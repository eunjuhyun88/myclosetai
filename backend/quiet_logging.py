#!/usr/bin/env python3
"""
로그 레벨 조정 스크립트 - 불필요한 상세 정보 숨기기
"""

import logging
import os

def setup_quiet_logging():
    """불필요한 상세 로그를 숨기는 로깅 설정"""
    
    # 로그 레벨 설정
    logging.basicConfig(
        level=logging.INFO,  # INFO 레벨로 설정 (DEBUG 메시지 숨김)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 특정 모듈들의 로그 레벨 조정
    quiet_modules = [
        'app.ai_pipeline.steps.step_01_human_parsing',
        'app.ai_pipeline.steps.step_02_pose_estimation', 
        'app.ai_pipeline.steps.step_03_cloth_segmentation',
        'app.ai_pipeline.steps.step_04_geometric_matching',
        'app.ai_pipeline.steps.step_05_cloth_warping',
        'app.ai_pipeline.steps.step_06_virtual_fitting',
        'app.ai_pipeline.steps.step_07_post_processing',
        'app.ai_pipeline.steps.step_08_quality_assessment',
        'app.ai_pipeline.utils.model_loader',
        'app.core.di_container',
        'app.services.step_service'
    ]
    
    for module in quiet_modules:
        logger = logging.getLogger(module)
        logger.setLevel(logging.INFO)  # DEBUG 메시지 숨김
    
    # 체크포인트 키 관련 로그를 DEBUG 레벨로 설정
    checkpoint_loggers = [
        'steps.HumanParsingStep',
        'steps.PoseEstimationStep',
        'steps.ClothSegmentationStep',
        'steps.GeometricMatchingStep',
        'steps.ClothWarpingStep',
        'steps.VirtualFittingStep',
        'steps.PostProcessingStep',
        'steps.QualityAssessmentStep'
    ]
    
    for logger_name in checkpoint_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)  # DEBUG 메시지 숨김
    
    print("✅ 로그 레벨 조정 완료 - 불필요한 상세 정보 숨김")

if __name__ == "__main__":
    setup_quiet_logging() 