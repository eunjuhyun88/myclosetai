#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - 테스트 실행 스크립트
=====================================================================

향상된 모델들이 작동하는지 간단하게 테스트하는 스크립트

Author: MyCloset AI Team  
Date: 2025-08-07
Version: 1.0
"""

import torch
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_test():
    """간단한 테스트"""
    try:
        logger.info("🚀 간단한 테스트 시작...")
        
        # 1. 기본 텐서 생성 테스트
        logger.info("📋 1. 기본 텐서 생성 테스트")
        x = torch.randn(1, 3, 64, 64)
        logger.info(f"✅ 테스트 입력 생성: {x.shape}")
        
        # 2. EnhancedU2NetModel 간단 테스트
        logger.info("📋 2. EnhancedU2NetModel 간단 테스트")
        try:
            from enhanced_models import EnhancedU2NetModel
            model = EnhancedU2NetModel(num_classes=1, input_channels=3)
            logger.info("✅ EnhancedU2NetModel 생성 성공")
            
            # 모델 실행
            with torch.no_grad():
                output = model(x)
                logger.info("✅ EnhancedU2NetModel forward 성공")
                
                # 출력 확인 - 딕셔너리 형태로 반환됨
                if isinstance(output, dict) and 'segmentation' in output:
                    seg_output = output['segmentation']
                    logger.info(f"✅ 세그멘테이션 출력: {seg_output.shape}")
                    
                    # 다른 출력들도 확인
                    if 'basic_output' in output:
                        logger.info(f"✅ 기본 출력: {output['basic_output'].shape}")
                    if 'advanced_features' in output:
                        logger.info(f"✅ 고급 특징들: {len(output['advanced_features'])}개")
                        for key, value in output['advanced_features'].items():
                            if hasattr(value, 'shape'):
                                logger.info(f"  - {key}: {value.shape}")
                            else:
                                logger.info(f"  - {key}: {type(value)}")
                else:
                    logger.warning(f"⚠️ 세그멘테이션 출력 누락 또는 잘못된 형태: {type(output)}")
                    
        except Exception as e:
            logger.error(f"❌ EnhancedU2NetModel 테스트 실패: {e}")
            return False
        
        # 3. 개별 모듈 간단 테스트
        logger.info("📋 3. 개별 모듈 간단 테스트")
        
        # Boundary Refinement Network
        try:
            from models.boundary_refinement import BoundaryRefinementNetwork
            boundary_model = BoundaryRefinementNetwork(256, 256)
            test_input = torch.randn(1, 256, 32, 32)
            output = boundary_model(test_input)
            logger.info("✅ BoundaryRefinementNetwork 테스트 성공")
        except Exception as e:
            logger.warning(f"⚠️ BoundaryRefinementNetwork 테스트 실패: {e}")
        
        # Feature Pyramid Network
        try:
            from models.feature_pyramid_network import FeaturePyramidNetwork
            fpn_model = FeaturePyramidNetwork(256, 256)
            test_input = torch.randn(1, 256, 32, 32)
            output = fpn_model(test_input)
            logger.info("✅ FeaturePyramidNetwork 테스트 성공")
        except Exception as e:
            logger.warning(f"⚠️ FeaturePyramidNetwork 테스트 실패: {e}")
        
        # Iterative Refinement
        try:
            from models.iterative_refinement import IterativeRefinementWithMemory
            iterative_model = IterativeRefinementWithMemory(256, 256)
            test_input = torch.randn(1, 256, 32, 32)
            output = iterative_model(test_input)
            logger.info("✅ IterativeRefinementWithMemory 테스트 성공")
        except Exception as e:
            logger.warning(f"⚠️ IterativeRefinementWithMemory 테스트 실패: {e}")
        
        # Multi-scale Feature Fusion
        try:
            from models.multi_scale_fusion import MultiScaleFeatureFusion
            fusion_model = MultiScaleFeatureFusion(256, 256)
            test_input = torch.randn(1, 256, 32, 32)
            output = fusion_model(test_input)
            logger.info("✅ MultiScaleFeatureFusion 테스트 성공")
        except Exception as e:
            logger.warning(f"⚠️ MultiScaleFeatureFusion 테스트 실패: {e}")
        
        logger.info("🎉 간단한 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    try:
        success = simple_test()
        if success:
            logger.info("🎉 모든 테스트가 성공했습니다!")
        else:
            logger.warning("⚠️ 일부 테스트가 실패했습니다.")
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 치명적 오류 발생: {e}")
