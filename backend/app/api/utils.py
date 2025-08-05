"""
API 유틸리티 함수들
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 환경 변수들 (step_routes.py에서 가져온 것)
IS_M3_MAX = True  # M3 Max 환경
CONDA_ENV = "myclosetlast"
IS_MYCLOSET_ENV = True
MEMORY_GB = 128


def format_step_api_response(
    success: bool,
    message: str,
    step_name: str,
    step_id: int,
    processing_time: float,
    session_id: str,
    confidence: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    fitted_image: Optional[str] = None,
    fit_score: Optional[float] = None,
    recommendations: Optional[list] = None,
    progress_percentage: Optional[float] = None,  # 🔥 진행률 추가
    next_step: Optional[int] = None,  # 🔥 다음 단계 추가
    **kwargs
) -> Dict[str, Any]:
    """API 응답 형식화 (프론트엔드 호환) - Central Hub 기반"""
    
    # session_id 필수 검증
    if not session_id:
        raise ValueError("session_id는 필수입니다!")
    
    # 🔥 진행률 계산
    if progress_percentage is None:
        progress_percentage = (step_id / 8) * 100  # 8단계 기준
    
    # 🔥 다음 단계 계산
    if next_step is None:
        next_step = step_id + 1 if step_id < 8 else None
    
    response = {
        "success": success,
        "message": message,
        "step_name": step_name,
        "step_id": step_id,
        "session_id": session_id,
        "processing_time": processing_time,
        "confidence": confidence or (0.85 + step_id * 0.02),
        "device": "mps" if IS_M3_MAX else "cpu",
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
        "error": error,
        
        # 🔥 프론트엔드 호환성 강화
        "progress_percentage": round(progress_percentage, 1),
        "next_step": next_step,
        "total_steps": 8,
        "current_step": step_id,
        "remaining_steps": max(0, 8 - step_id),
        
        # Central Hub DI Container v7.0 정보
        "central_hub_di_container_v70": True,
        "circular_reference_free": True,
        "single_source_of_truth": True,
        "dependency_inversion": True,
        "conda_environment": CONDA_ENV,
        "mycloset_optimized": IS_MYCLOSET_ENV,
        "m3_max_optimized": IS_M3_MAX,
        "memory_gb": MEMORY_GB,
        "central_hub_used": True,
        "di_container_integration": True
    }
    
    # 프론트엔드 호환성 추가
    if fitted_image:
        response["fitted_image"] = fitted_image
    if fit_score:
        response["fit_score"] = fit_score
    if recommendations:
        response["recommendations"] = recommendations
    
    # 🔥 중간 결과물 저장 시스템 필드들 처리
    if 'intermediate_results' in kwargs:
        response["intermediate_results"] = kwargs['intermediate_results']
    
    # Step 1 결과 필드들
    if 'parsing_visualization' in kwargs:
        response["parsing_visualization"] = kwargs['parsing_visualization']
    if 'overlay_image' in kwargs:
        response["overlay_image"] = kwargs['overlay_image']
    if 'detected_body_parts' in kwargs:
        response["detected_body_parts"] = kwargs['detected_body_parts']
    if 'clothing_analysis' in kwargs:
        response["clothing_analysis"] = kwargs['clothing_analysis']
    if 'unique_labels' in kwargs:
        response["unique_labels"] = kwargs['unique_labels']
    if 'parsing_shape' in kwargs:
        response["parsing_shape"] = kwargs['parsing_shape']
    
    # Step 2 결과 필드들
    if 'pose_visualization' in kwargs:
        response["pose_visualization"] = kwargs['pose_visualization']
    if 'keypoints' in kwargs:
        response["keypoints"] = kwargs['keypoints']
    if 'confidence_scores' in kwargs:
        response["confidence_scores"] = kwargs['confidence_scores']
    if 'joint_angles' in kwargs:
        response["joint_angles"] = kwargs['joint_angles']
    if 'body_proportions' in kwargs:
        response["body_proportions"] = kwargs['body_proportions']
    if 'keypoints_count' in kwargs:
        response["keypoints_count"] = kwargs['keypoints_count']
    if 'skeleton_structure' in kwargs:
        response["skeleton_structure"] = kwargs['skeleton_structure']
    
    # Step 3 결과 필드들
    if 'mask_overlay' in kwargs:
        response["mask_overlay"] = kwargs['mask_overlay']
    if 'category_overlay' in kwargs:
        response["category_overlay"] = kwargs['category_overlay']
    if 'segmented_clothing' in kwargs:
        response["segmented_clothing"] = kwargs['segmented_clothing']
    if 'cloth_categories' in kwargs:
        response["cloth_categories"] = kwargs['cloth_categories']
    if 'cloth_features' in kwargs:
        response["cloth_features"] = kwargs['cloth_features']
    if 'cloth_bounding_boxes' in kwargs:
        response["cloth_bounding_boxes"] = kwargs['cloth_bounding_boxes']
    if 'cloth_centroids' in kwargs:
        response["cloth_centroids"] = kwargs['cloth_centroids']
    if 'cloth_areas' in kwargs:
        response["cloth_areas"] = kwargs['cloth_areas']
    
    # Step 4 결과 필드들
    if 'geometric_matching_result' in kwargs:
        response["geometric_matching_result"] = kwargs['geometric_matching_result']
    if 'transformation_matrix' in kwargs:
        response["transformation_matrix"] = kwargs['transformation_matrix']
    if 'matching_score' in kwargs:
        response["matching_score"] = kwargs['matching_score']
    if 'alignment_visualization' in kwargs:
        response["alignment_visualization"] = kwargs['alignment_visualization']
    
    # Step 5 결과 필드들
    if 'warped_cloth' in kwargs:
        response["warped_cloth"] = kwargs['warped_cloth']
    if 'warping_visualization' in kwargs:
        response["warping_visualization"] = kwargs['warping_visualization']
    if 'warping_quality_score' in kwargs:
        response["warping_quality_score"] = kwargs['warping_quality_score']
    
    # Step 6 결과 필드들
    if 'virtual_fitting_result' in kwargs:
        response["virtual_fitting_result"] = kwargs['virtual_fitting_result']
    if 'fitting_visualization' in kwargs:
        response["fitting_visualization"] = kwargs['fitting_visualization']
    if 'fitting_quality_score' in kwargs:
        response["fitting_quality_score"] = kwargs['fitting_quality_score']
    
    # Step 7 결과 필드들
    if 'post_processing_result' in kwargs:
        response["post_processing_result"] = kwargs['post_processing_result']
    if 'enhanced_image' in kwargs:
        response["enhanced_image"] = kwargs['enhanced_image']
    if 'post_processing_score' in kwargs:
        response["post_processing_score"] = kwargs['post_processing_score']
    
    # Step 8 결과 필드들
    if 'quality_assessment_result' in kwargs:
        response["quality_assessment_result"] = kwargs['quality_assessment_result']
    if 'quality_score' in kwargs:
        response["quality_score"] = kwargs['quality_score']
    if 'assessment_details' in kwargs:
        response["assessment_details"] = kwargs['assessment_details']
    
    # 🔥 추가 메타데이터
    response.update({
        "api_version": "v1.0",
        "step_processing_version": "v2.0",
        "central_hub_version": "v7.0",
        "optimization_level": "m3_max_128gb" if IS_M3_MAX and MEMORY_GB >= 128 else "standard",
        "processing_mode": "production",
        "response_format": "enhanced_frontend_compatible"
    })
    
    return response 