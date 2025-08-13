#!/usr/bin/env python3
"""
🔥 DeepLabV3+ 디버그 테스트
=====================================================================

DeepLabV3+ 모델의 문제를 정확히 파악하기 위한 테스트

Author: MyCloset AI Team  
Date: 2025-08-01
"""

import torch
import torch.nn as nn
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deeplabv3plus_step_by_step():
    """DeepLabV3+ 단계별 테스트"""
    try:
        logger.info("🔍 DeepLabV3+ 단계별 테스트 시작")
        
        from models.deeplabv3plus import DeepLabV3PlusModel
        
        # 모델 생성
        model = DeepLabV3PlusModel(num_classes=2)
        logger.info("✅ DeepLabV3+ 모델 생성 성공")
        
        # 테스트 입력
        test_input = torch.randn(1, 3, 256, 256)
        logger.info(f"✅ 테스트 입력 생성: {test_input.shape}")
        
        # Backbone 테스트
        try:
            low_level_feat, high_level_feat = model.backbone(test_input)
            logger.info(f"✅ Backbone 성공 - Low: {low_level_feat.shape}, High: {high_level_feat.shape}")
        except Exception as e:
            logger.error(f"❌ Backbone 실패: {e}")
            return False
        
        # ASPP 테스트
        try:
            aspp_feat = model.aspp(high_level_feat)
            logger.info(f"✅ ASPP 성공 - 출력: {aspp_feat.shape}")
        except Exception as e:
            logger.error(f"❌ ASPP 실패: {e}")
            return False
        
        # Decoder 테스트
        try:
            output = model.decoder(aspp_feat, low_level_feat, (256, 256))
            logger.info(f"✅ Decoder 성공 - 출력: {output.shape}")
        except Exception as e:
            logger.error(f"❌ Decoder 실패: {e}")
            return False
        
        # 전체 모델 테스트
        try:
            full_output = model(test_input)
            logger.info(f"✅ 전체 모델 성공 - 출력: {full_output.shape}")
        except Exception as e:
            logger.error(f"❌ 전체 모델 실패: {e}")
            return False
        
        logger.info("✅ DeepLabV3+ 단계별 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ DeepLabV3+ 단계별 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_deeplabv3plus_step_by_step()
