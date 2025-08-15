#!/usr/bin/env python3
"""
🔥 Quality Assessment 설정 파일
================================================================================

✅ 품질 평가 설정
✅ 모델 파라미터
✅ 메트릭 가중치
✅ 시스템 설정

Author: MyCloset AI Team
Date: 2025-08-13
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class QualityLevel(Enum):
    """품질 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class AssessmentType(Enum):
    """평가 타입"""
    ABSOLUTE = "absolute"      # 절대적 품질 평가
    RELATIVE = "relative"      # 상대적 품질 평가
    COMPARATIVE = "comparative" # 비교 품질 평가
    COMPREHENSIVE = "comprehensive" # 종합 품질 평가

class MetricType(Enum):
    """메트릭 타입"""
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
    FID = "fid"
    MAE = "mae"
    RMSE = "rmse"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    COLOR_CONSISTENCY = "color_consistency"
    TEXTURE_PRESERVATION = "texture_preservation"

class AssessmentQuality(Enum):
    """평가 품질"""
    EXCELLENT = "excellent"     # 90-100점
    GOOD = "good"              # 75-89점
    ACCEPTABLE = "acceptable"   # 60-74점
    POOR = "poor"              # 40-59점
    VERY_POOR = "very_poor"    # 0-39점

@dataclass
class AssessmentResult:
    """평가 결과"""
    overall_score: float = 0.0
    quality_grade: str = "C"
    confidence: float = 0.0
    assessment_type: AssessmentType = AssessmentType.COMPREHENSIVE
    quality_level: AssessmentQuality = AssessmentQuality.ACCEPTABLE
    
    # 세부 메트릭
    psnr_score: float = 0.0
    ssim_score: float = 0.0
    lpips_score: float = 0.0
    mae_score: float = 0.0
    rmse_score: float = 0.0
    
    # 품질 평가
    sharpness_score: float = 0.0
    color_score: float = 0.0
    fitting_score: float = 0.0
    realism_score: float = 0.0
    artifacts_score: float = 0.0
    
    # 메타데이터
    timestamp: str = ""
    model_version: str = ""
    processing_time: float = 0.0
    
    def get_overall_quality(self) -> str:
        """전체 품질 등급 반환"""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "A-"
        elif self.overall_score >= 0.80:
            return "B+"
        elif self.overall_score >= 0.75:
            return "B"
        elif self.overall_score >= 0.70:
            return "B-"
        elif self.overall_score >= 0.65:
            return "C+"
        elif self.overall_score >= 0.60:
            return "C"
        else:
            return "D"

# 품질 평가 상수들
ASSESSMENT_CONSTANTS = {
    'DEFAULT_WEIGHTS': {
        'psnr': 0.2,
        'ssim': 0.3,
        'lpips': 0.2,
        'mae': 0.1,
        'rmse': 0.1,
        'sharpness': 0.05,
        'color': 0.05
    },
    
    'QUALITY_THRESHOLDS': {
        'excellent': 0.90,
        'good': 0.75,
        'acceptable': 0.60,
        'poor': 0.40,
        'very_poor': 0.0
    },
    
    'METRIC_RANGES': {
        'psnr': {'min': 0.0, 'max': 100.0, 'optimal': 50.0},
        'ssim': {'min': 0.0, 'max': 1.0, 'optimal': 0.9},
        'lpips': {'min': 0.0, 'max': 1.0, 'optimal': 0.1},
        'mae': {'min': 0.0, 'max': 255.0, 'optimal': 0.0},
        'rmse': {'min': 0.0, 'max': 255.0, 'optimal': 0.0}
    },
    
    'PROCESSING_TIMEOUTS': {
        'fast': 5.0,      # 5초
        'normal': 15.0,   # 15초
        'thorough': 30.0  # 30초
    }
}

@dataclass
class ModelConfig:
    """모델 설정"""
    # Transformer 모델 설정
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Cross-Attention 모델 설정
    cross_attention_d_model: int = 256
    cross_attention_num_heads: int = 8
    
    # 앙상블 설정
    ensemble_enabled: bool = True
    ensemble_weights_learnable: bool = True
    
    # 품질 헤드 설정
    quality_heads: List[str] = field(default_factory=lambda: [
        'overall', 'sharpness', 'color', 'fitting', 'realism', 'artifacts'
    ])

@dataclass
class MetricsConfig:
    """메트릭 설정"""
    # 기본 메트릭 가중치
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'psnr': 0.25,           # PSNR 가중치 25%
        'ssim': 0.25,           # SSIM 가중치 25%
        'lpips': 0.15,          # LPIPS 가중치 15%
        'mae': 0.10,            # MAE 가중치 10%
        'rmse': 0.10,           # RMSE 가중치 10%
        'structural_similarity': 0.05,  # 구조적 유사성 5%
        'color_consistency': 0.05,      # 색상 일관성 5%
        'texture_preservation': 0.05    # 텍스처 보존도 5%
    })
    
    # 품질 등급 임계값
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'A+': 0.95,
        'A': 0.90,
        'A-': 0.85,
        'B+': 0.80,
        'B': 0.75,
        'B-': 0.70,
        'C+': 0.65,
        'C': 0.60,
        'C-': 0.55,
        'D+': 0.50,
        'D': 0.45,
        'F': 0.00
    })
    
    # 메트릭별 임계값
    metric_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'psnr': {'excellent': 40.0, 'good': 30.0, 'poor': 20.0},
        'ssim': {'excellent': 0.95, 'good': 0.90, 'poor': 0.80},
        'lpips': {'excellent': 0.10, 'good': 0.20, 'poor': 0.30},
        'fid': {'excellent': 10.0, 'good': 20.0, 'poor': 30.0}
    })

@dataclass
class PerformanceConfig:
    """성능 설정"""
    # 장치 설정
    device: str = "auto"  # auto, cpu, cuda, mps
    device_preference: List[str] = field(default_factory=lambda: ["mps", "cuda", "cpu"])
    
    # 배치 처리 설정
    batch_size: int = 1
    max_batch_size: int = 8
    
    # 메모리 설정
    memory_efficient: bool = True
    gradient_checkpointing: bool = False
    
    # 추론 설정
    inference_mode: bool = True
    use_amp: bool = False  # Automatic Mixed Precision

@dataclass
class QualityAssessmentConfig:
    """품질 평가 전체 설정"""
    
    # 기본 설정
    name: str = "Quality Assessment Step"
    version: str = "1.0"
    description: str = "고급 신경망 기반 품질 평가 시스템"
    
    # 평가 설정
    assessment_type: AssessmentType = AssessmentType.COMPREHENSIVE
    quality_level: QualityLevel = QualityLevel.HIGH
    
    # 모델 설정
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 메트릭 설정
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # 성능 설정
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # 로깅 설정
    logging_level: str = "INFO"
    save_intermediate_results: bool = False
    save_quality_report: bool = True
    
    # 출력 설정
    output_format: str = "json"  # json, xml, yaml
    include_visualizations: bool = True
    include_detailed_metrics: bool = True
    
    # 자동 후처리 설정
    auto_postprocessing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'assessment_type': self.assessment_type.value,
            'quality_level': self.quality_level.value,
            'model': {
                'd_model': self.model.d_model,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'd_ff': self.model.d_ff,
                'dropout': self.model.dropout,
                'cross_attention_d_model': self.model.cross_attention_d_model,
                'cross_attention_num_heads': self.model.cross_attention_num_heads,
                'ensemble_enabled': self.model.ensemble_enabled,
                'ensemble_weights_learnable': self.model.ensemble_weights_learnable,
                'quality_heads': self.model.quality_heads
            },
            'metrics': {
                'metric_weights': self.metrics.metric_weights,
                'grade_thresholds': self.metrics.grade_thresholds,
                'metric_thresholds': self.metrics.metric_thresholds
            },
            'performance': {
                'device': self.performance.device,
                'device_preference': self.performance.device_preference,
                'batch_size': self.performance.batch_size,
                'max_batch_size': self.performance.max_batch_size,
                'memory_efficient': self.performance.memory_efficient,
                'gradient_checkpointing': self.performance.gradient_checkpointing,
                'inference_mode': self.performance.inference_mode,
                'use_amp': self.performance.use_amp
            },
            'logging_level': self.logging_level,
            'save_intermediate_results': self.save_intermediate_results,
            'save_quality_report': self.save_quality_report,
            'output_format': self.output_format,
            'include_visualizations': self.include_visualizations,
            'include_detailed_metrics': self.include_detailed_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QualityAssessmentConfig':
        """딕셔너리에서 설정 생성"""
        config = cls()
        
        # 기본 설정
        if 'name' in config_dict:
            config.name = config_dict['name']
        if 'version' in config_dict:
            config.version = config_dict['version']
        if 'description' in config_dict:
            config.description = config_dict['description']
        
        # 평가 설정
        if 'assessment_type' in config_dict:
            config.assessment_type = AssessmentType(config_dict['assessment_type'])
        if 'quality_level' in config_dict:
            config.quality_level = QualityLevel(config_dict['quality_level'])
        
        # 모델 설정
        if 'model' in config_dict:
            model_config = config_dict['model']
            if 'd_model' in model_config:
                config.model.d_model = model_config['d_model']
            if 'num_layers' in model_config:
                config.model.num_layers = model_config['num_layers']
            if 'num_heads' in model_config:
                config.model.num_heads = model_config['num_heads']
            if 'd_ff' in model_config:
                config.model.d_ff = model_config['d_ff']
            if 'dropout' in model_config:
                config.model.dropout = model_config['dropout']
            if 'cross_attention_d_model' in model_config:
                config.model.cross_attention_d_model = model_config['cross_attention_d_model']
            if 'cross_attention_num_heads' in model_config:
                config.model.cross_attention_num_heads = model_config['cross_attention_num_heads']
            if 'ensemble_enabled' in model_config:
                config.model.ensemble_enabled = model_config['ensemble_enabled']
            if 'ensemble_weights_learnable' in model_config:
                config.model.ensemble_weights_learnable = model_config['ensemble_weights_learnable']
            if 'quality_heads' in model_config:
                config.model.quality_heads = model_config['quality_heads']
        
        # 메트릭 설정
        if 'metrics' in config_dict:
            metrics_config = config_dict['metrics']
            if 'metric_weights' in metrics_config:
                config.metrics.metric_weights = metrics_config['metric_weights']
            if 'grade_thresholds' in metrics_config:
                config.metrics.grade_thresholds = metrics_config['grade_thresholds']
            if 'metric_thresholds' in metrics_config:
                config.metrics.metric_thresholds = metrics_config['metric_thresholds']
        
        # 성능 설정
        if 'performance' in config_dict:
            perf_config = config_dict['performance']
            if 'device' in perf_config:
                config.performance.device = perf_config['device']
            if 'device_preference' in perf_config:
                config.performance.device_preference = perf_config['device_preference']
            if 'batch_size' in perf_config:
                config.performance.batch_size = perf_config['batch_size']
            if 'max_batch_size' in perf_config:
                config.performance.max_batch_size = perf_config['max_batch_size']
            if 'memory_efficient' in perf_config:
                config.performance.memory_efficient = perf_config['memory_efficient']
            if 'gradient_checkpointing' in perf_config:
                config.performance.gradient_checkpointing = perf_config['gradient_checkpointing']
            if 'inference_mode' in perf_config:
                config.performance.inference_mode = perf_config['inference_mode']
            if 'use_amp' in perf_config:
                config.performance.use_amp = perf_config['use_amp']
        
        # 기타 설정
        if 'logging_level' in config_dict:
            config.logging_level = config_dict['logging_level']
        if 'save_intermediate_results' in config_dict:
            config.save_intermediate_results = config_dict['save_intermediate_results']
        if 'save_quality_report' in config_dict:
            config.save_quality_report = config_dict['save_quality_report']
        if 'output_format' in config_dict:
            config.output_format = config_dict['output_format']
        if 'include_visualizations' in config_dict:
            config.include_visualizations = config_dict['include_visualizations']
        if 'include_detailed_metrics' in config_dict:
            config.include_detailed_metrics = config_dict['include_detailed_metrics']
        
        return config

# 기본 설정
DEFAULT_CONFIG = QualityAssessmentConfig()

# 고품질 설정
HIGH_QUALITY_CONFIG = QualityAssessmentConfig(
    quality_level=QualityLevel.HIGH,
    model=ModelConfig(
        d_model=768,
        num_layers=8,
        num_heads=12,
        d_ff=3072,
        dropout=0.1
    ),
    performance=PerformanceConfig(
        use_amp=True,
        memory_efficient=True
    )
)

# 초고품질 설정
ULTRA_QUALITY_CONFIG = QualityAssessmentConfig(
    quality_level=QualityLevel.ULTRA,
    model=ModelConfig(
        d_model=1024,
        num_layers=12,
        num_heads=16,
        d_ff=4096,
        dropout=0.1
    ),
    performance=PerformanceConfig(
        use_amp=True,
        memory_efficient=True,
        gradient_checkpointing=True
    )
)

# 모델 설정들
MODEL_CONFIGS = {
    'default': DEFAULT_CONFIG,
    'high_quality': HIGH_QUALITY_CONFIG,
    'ultra_quality': ULTRA_QUALITY_CONFIG
}

# 품질 평가 모델 클래스 (Mock)
class QualityAssessmentModel:
    """품질 평가 모델 (Mock)"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.name = "QualityAssessmentModel"
        self.version = "1.0.0"
    
    def evaluate(self, image, reference=None):
        """이미지 품질 평가"""
        return {
            'quality_score': 0.85,
            'grade': 'B+',
            'details': {
                'sharpness': 0.8,
                'color': 0.9,
                'composition': 0.85
            }
        }

# 품질 평가 유틸리티 클래스 (Mock)
class QualityAssessmentUtils:
    """품질 평가 유틸리티 (Mock)"""
    
    @staticmethod
    def calculate_psnr(image1, image2):
        """PSNR 계산"""
        return 30.5
    
    @staticmethod
    def calculate_ssim(image1, image2):
        """SSIM 계산"""
        return 0.85
    
    @staticmethod
    def calculate_lpips(image1, image2):
        """LPIPS 계산"""
        return 0.15
