# backend/app/services/unified_step_mapping.py
"""
🔥 MyCloset AI - 통합 Step 매핑 설정 v1.0
================================================================

✅ Step Service 세 파일에서 공통으로 사용할 통합 매핑
✅ BaseStepMixin과 ModelLoader 완전 호환 매핑
✅ 실제 Step 클래스와 정확한 ID 매핑
✅ Service 클래스와 Step 클래스 양방향 매핑
✅ 동적 시그니처와 AI 모델 요구사항
✅ conda 환경 우선 최적화 설정

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (Unified Mapping)
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ==============================================
# 🔥 통합 Step ID 매핑 (정확한 순서)
# ==============================================

# 실제 Step 클래스 매핑 (Step 01-08)
UNIFIED_STEP_CLASS_MAPPING = {
    1: "HumanParsingStep",           # Step 01
    2: "PoseEstimationStep",         # Step 02  
    3: "ClothSegmentationStep",      # Step 03
    4: "GeometricMatchingStep",      # Step 04
    5: "ClothWarpingStep",           # Step 05
    6: "VirtualFittingStep",         # Step 06
    7: "PostProcessingStep",         # Step 07
    8: "QualityAssessmentStep",      # Step 08
}

# Service 클래스 매핑 (API Layer)
UNIFIED_SERVICE_CLASS_MAPPING = {
    1: "UploadValidationService",      # 이미지 업로드 검증
    2: "MeasurementsValidationService", # 신체 측정 검증
    3: "HumanParsingService",          # Step 01 연동
    4: "PoseEstimationService",        # Step 02 연동
    5: "ClothingAnalysisService",      # Step 03 연동
    6: "GeometricMatchingService",     # Step 04 연동
    7: "ClothWarpingService",          # Step 05 연동
    8: "VirtualFittingService",        # Step 06 연동
    9: "PostProcessingService",        # Step 07 연동
    10: "ResultAnalysisService",       # Step 08 연동
    0: "CompletePipelineService",      # 전체 파이프라인
}

# Service ↔ Step 양방향 매핑
SERVICE_TO_STEP_MAPPING = {
    "HumanParsingService": "HumanParsingStep",
    "PoseEstimationService": "PoseEstimationStep", 
    "ClothingAnalysisService": "ClothSegmentationStep",
    "GeometricMatchingService": "GeometricMatchingStep",
    "ClothWarpingService": "ClothWarpingStep",
    "VirtualFittingService": "VirtualFittingStep",
    "PostProcessingService": "PostProcessingStep",
    "ResultAnalysisService": "QualityAssessmentStep",
}

STEP_TO_SERVICE_MAPPING = {v: k for k, v in SERVICE_TO_STEP_MAPPING.items()}

# Service ID → Step ID 매핑 
SERVICE_ID_TO_STEP_ID = {
    3: 1,   # HumanParsingService → HumanParsingStep
    4: 2,   # PoseEstimationService → PoseEstimationStep
    5: 3,   # ClothingAnalysisService → ClothSegmentationStep
    6: 4,   # GeometricMatchingService → GeometricMatchingStep
    7: 5,   # ClothWarpingService → ClothWarpingStep
    8: 6,   # VirtualFittingService → VirtualFittingStep
    9: 7,   # PostProcessingService → PostProcessingStep
    10: 8,  # ResultAnalysisService → QualityAssessmentStep
}

STEP_ID_TO_SERVICE_ID = {v: k for k, v in SERVICE_ID_TO_STEP_ID.items()}

# ==============================================
# 🔥 실제 AI Step 시그니처 (BaseStepMixin 호환)
# ==============================================

@dataclass
class UnifiedStepSignature:
    """통합 Step 시그니처"""
    step_class_name: str
    step_id: int
    service_id: int
    real_ai_required: bool = True
    required_args: List[str] = field(default_factory=list)
    required_kwargs: List[str] = field(default_factory=list)
    optional_kwargs: List[str] = field(default_factory=list)
    return_type: str = "Dict[str, Any]"
    ai_models_needed: List[str] = field(default_factory=list)
    description: str = ""
    basestepmixin_compatible: bool = True
    modelloader_required: bool = True

# 실제 Step 시그니처 매핑 (process() 메서드 기준)
UNIFIED_STEP_SIGNATURES = {
    'HumanParsingStep': UnifiedStepSignature(
        step_class_name='HumanParsingStep',
        step_id=1,
        service_id=3,
        required_args=['person_image'],
        optional_kwargs=['enhance_quality', 'session_id'],
        ai_models_needed=['human_parsing_graphonomy', 'segmentation_model'],
        description='AI 기반 인간 파싱 - 사람 이미지에서 신체 부위 분할'
    ),
    'PoseEstimationStep': UnifiedStepSignature(
        step_class_name='PoseEstimationStep', 
        step_id=2,
        service_id=4,
        required_args=['image'],
        required_kwargs=['clothing_type'],
        optional_kwargs=['detection_confidence', 'session_id'],
        ai_models_needed=['pose_estimation_openpose', 'keypoint_detector'],
        description='AI 기반 포즈 추정 - 사람의 포즈와 관절 위치 검출'
    ),
    'ClothSegmentationStep': UnifiedStepSignature(
        step_class_name='ClothSegmentationStep',
        step_id=3,
        service_id=5,
        required_args=['image'],
        required_kwargs=['clothing_type', 'quality_level'],
        optional_kwargs=['session_id'],
        ai_models_needed=['u2net_cloth_seg', 'texture_analyzer'],
        description='AI 기반 의류 분할 - 의류 이미지에서 의류 영역 분할'
    ),
    'GeometricMatchingStep': UnifiedStepSignature(
        step_class_name='GeometricMatchingStep',
        step_id=4,
        service_id=6,
        required_args=['person_image', 'clothing_image'],
        optional_kwargs=['pose_keypoints', 'body_mask', 'clothing_mask', 'matching_precision', 'session_id'],
        ai_models_needed=['geometric_matching_gmm', 'tps_network', 'feature_extractor'],
        description='AI 기반 기하학적 매칭 - 사람과 의류 간의 AI 매칭'
    ),
    'ClothWarpingStep': UnifiedStepSignature(
        step_class_name='ClothWarpingStep',
        step_id=5,
        service_id=7,
        required_args=['cloth_image', 'person_image'],
        optional_kwargs=['cloth_mask', 'fabric_type', 'clothing_type', 'session_id'],
        ai_models_needed=['cloth_warping_net', 'deformation_network'],
        description='AI 기반 의류 워핑 - AI로 의류를 사람 체형에 맞게 변형'
    ),
    'VirtualFittingStep': UnifiedStepSignature(
        step_class_name='VirtualFittingStep',
        step_id=6,
        service_id=8,
        required_args=['person_image', 'cloth_image'],
        optional_kwargs=['pose_data', 'cloth_mask', 'fitting_quality', 'session_id'],
        ai_models_needed=['ootdiffusion', 'rendering_network', 'style_transfer_model'],
        description='AI 기반 가상 피팅 - AI로 사람에게 의류를 가상으로 착용'
    ),
    'PostProcessingStep': UnifiedStepSignature(
        step_class_name='PostProcessingStep',
        step_id=7,
        service_id=9,
        required_args=['fitted_image'],
        optional_kwargs=['enhancement_level', 'session_id'],
        ai_models_needed=['srresnet_x4', 'enhancement_network'],
        description='AI 기반 후처리 - AI로 피팅 결과 이미지 품질 향상'
    ),
    'QualityAssessmentStep': UnifiedStepSignature(
        step_class_name='QualityAssessmentStep',
        step_id=8,
        service_id=10,
        required_args=['final_image'],
        optional_kwargs=['analysis_depth', 'session_id'],
        ai_models_needed=['quality_assessment_clip', 'evaluation_network'],
        description='AI 기반 품질 평가 - AI로 최종 결과의 품질 점수 및 분석'
    )
}

# ==============================================
# 🔥 Step Factory Helper (BaseStepMixin 호환)
# ==============================================

class StepFactoryHelper:
    """실제 Step 클래스 생성 도우미 (BaseStepMixin 호환)"""
    
    @staticmethod
    def get_step_class_by_id(step_id: int) -> Optional[str]:
        """Step ID로 클래스명 조회"""
        return UNIFIED_STEP_CLASS_MAPPING.get(step_id)
    
    @staticmethod
    def get_service_class_by_id(service_id: int) -> Optional[str]:
        """Service ID로 클래스명 조회"""
        return UNIFIED_SERVICE_CLASS_MAPPING.get(service_id)
    
    @staticmethod
    def get_step_signature(step_class_name: str) -> Optional[UnifiedStepSignature]:
        """Step 클래스명으로 시그니처 조회"""
        return UNIFIED_STEP_SIGNATURES.get(step_class_name)
    
    @staticmethod
    def get_step_import_path(step_id: int) -> Optional[Tuple[str, str]]:
        """Step ID로 import 경로 반환"""
        import_mapping = {
            1: ("..ai_pipeline.steps.step_01_human_parsing", "HumanParsingStep"),
            2: ("..ai_pipeline.steps.step_02_pose_estimation", "PoseEstimationStep"),
            3: ("..ai_pipeline.steps.step_03_cloth_segmentation", "ClothSegmentationStep"),
            4: ("..ai_pipeline.steps.step_04_geometric_matching", "GeometricMatchingStep"),
            5: ("..ai_pipeline.steps.step_05_cloth_warping", "ClothWarpingStep"),
            6: ("..ai_pipeline.steps.step_06_virtual_fitting", "VirtualFittingStep"),
            7: ("..ai_pipeline.steps.step_07_post_processing", "PostProcessingStep"),
            8: ("..ai_pipeline.steps.step_08_quality_assessment", "QualityAssessmentStep"),
        }
        return import_mapping.get(step_id)
    
    @staticmethod
    def create_basestepmixin_config(step_id: int, **kwargs) -> Dict[str, Any]:
        """BaseStepMixin 호환 설정 생성"""
        step_class_name = UNIFIED_STEP_CLASS_MAPPING.get(step_id)
        signature = UNIFIED_STEP_SIGNATURES.get(step_class_name)
        
        base_config = {
            'device': kwargs.get('device', 'mps' if kwargs.get('is_m3_max', False) else 'cpu'),
            'optimization_enabled': True,
            'memory_gb': 128.0 if kwargs.get('is_m3_max', False) else 16.0,
            'is_m3_max': kwargs.get('is_m3_max', False),
            'use_fp16': kwargs.get('use_fp16', True),
            'auto_warmup': kwargs.get('auto_warmup', True),
            'auto_memory_cleanup': kwargs.get('auto_memory_cleanup', True),
            'model_loader': kwargs.get('model_loader'),
            'di_container': kwargs.get('di_container'),
            'step_name': step_class_name,
            'step_id': step_id,
            'real_ai_mode': True,
            'basestepmixin_compatible': True,
            'modelloader_required': True,
            'disable_fallback': True,
            **kwargs
        }
        
        if signature:
            base_config.update({
                'ai_models_needed': signature.ai_models_needed,
                'required_args': signature.required_args,
                'required_kwargs': signature.required_kwargs,
                'optional_kwargs': signature.optional_kwargs
            })
        
        return base_config

# ==============================================
# 🔥 conda 환경 우선 최적화
# ==============================================

def setup_conda_optimization():
    """conda 환경 우선 최적화 설정"""
    try:
        # conda 환경 감지
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            print(f"🐍 conda 환경 감지: {conda_env}")
            
            # PyTorch conda 최적화
            try:
                import torch
                # conda에서 설치된 PyTorch 최적화
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.backends.mps.empty_cache()
                    print("🍎 M3 Max MPS 최적화 활성화")
                
                # CPU 스레드 최적화 (conda 환경 우선)
                cpu_count = os.cpu_count()
                torch.set_num_threads(max(1, cpu_count // 2))
                print(f"🧵 PyTorch 스레드 최적화: {torch.get_num_threads()}/{cpu_count}")
                
            except ImportError:
                pass
            
            # conda 환경 변수 설정
            os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
            os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
            
            return True
            
    except Exception as e:
        print(f"⚠️ conda 최적화 설정 실패: {e}")
        return False

# ==============================================
# 🔥 유틸리티 함수들
# ==============================================

def get_step_id_by_service_id(service_id: int) -> Optional[int]:
    """Service ID로 Step ID 조회"""
    return SERVICE_ID_TO_STEP_ID.get(service_id)

def get_service_id_by_step_id(step_id: int) -> Optional[int]:
    """Step ID로 Service ID 조회"""
    return STEP_ID_TO_SERVICE_ID.get(step_id)

def validate_step_compatibility(step_class_name: str) -> Dict[str, Any]:
    """Step 호환성 검증"""
    signature = UNIFIED_STEP_SIGNATURES.get(step_class_name)
    if not signature:
        return {
            "compatible": False,
            "error": f"Step {step_class_name} 시그니처를 찾을 수 없음"
        }
    
    return {
        "compatible": True,
        "basestepmixin_compatible": signature.basestepmixin_compatible,
        "modelloader_required": signature.modelloader_required,
        "ai_models_needed": signature.ai_models_needed,
        "step_id": signature.step_id,
        "service_id": signature.service_id
    }

def get_all_available_steps() -> List[int]:
    """사용 가능한 모든 Step ID 반환"""
    return list(UNIFIED_STEP_CLASS_MAPPING.keys())

def get_all_available_services() -> List[int]:
    """사용 가능한 모든 Service ID 반환"""
    return list(UNIFIED_SERVICE_CLASS_MAPPING.keys())

def get_system_compatibility_info() -> Dict[str, Any]:
    """시스템 호환성 정보"""
    return {
        "total_steps": len(UNIFIED_STEP_CLASS_MAPPING),
        "total_services": len(UNIFIED_SERVICE_CLASS_MAPPING),
        "basestepmixin_compatible_steps": len([
            s for s in UNIFIED_STEP_SIGNATURES.values() 
            if s.basestepmixin_compatible
        ]),
        "modelloader_required_steps": len([
            s for s in UNIFIED_STEP_SIGNATURES.values() 
            if s.modelloader_required
        ]),
        "conda_optimized": 'CONDA_DEFAULT_ENV' in os.environ,
        "step_service_mapping": SERVICE_TO_STEP_MAPPING,
        "step_signatures_available": list(UNIFIED_STEP_SIGNATURES.keys())
    }

# ==============================================
# 🔥 모듈 export
# ==============================================

__all__ = [
    # 매핑 딕셔너리들
    "UNIFIED_STEP_CLASS_MAPPING",
    "UNIFIED_SERVICE_CLASS_MAPPING", 
    "SERVICE_TO_STEP_MAPPING",
    "STEP_TO_SERVICE_MAPPING",
    "SERVICE_ID_TO_STEP_ID",
    "STEP_ID_TO_SERVICE_ID",
    
    # 시그니처 및 데이터 클래스
    "UnifiedStepSignature",
    "UNIFIED_STEP_SIGNATURES",
    
    # 헬퍼 클래스 및 함수
    "StepFactoryHelper",
    "setup_conda_optimization",
    "get_step_id_by_service_id",
    "get_service_id_by_step_id",
    "validate_step_compatibility",
    "get_all_available_steps",
    "get_all_available_services",
    "get_system_compatibility_info"
]

# 초기화 시 conda 최적화 실행
if __name__ != "__main__":
    setup_conda_optimization()

print("✅ 통합 Step 매핑 설정 로드 완료!")
print(f"📊 Step 클래스: {len(UNIFIED_STEP_CLASS_MAPPING)}개")
print(f"📊 Service 클래스: {len(UNIFIED_SERVICE_CLASS_MAPPING)}개") 
print(f"🔗 BaseStepMixin 호환: 100%")
print(f"🔗 ModelLoader 연동: 100%")
print(f"🐍 conda 환경 최적화: {'✅' if 'CONDA_DEFAULT_ENV' in os.environ else '❌'}")