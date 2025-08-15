#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - 통합 테스트
=====================================================================

의류 세그멘테이션 시스템의 모든 컴포넌트를 테스트
- 모델 팩토리 테스트
- 앙상블 시스템 테스트
- 통합 모델 관리자 테스트

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import sys
import os
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """모든 모듈 import 테스트"""
    logger.info("🔄 Import 테스트 시작...")
    
    try:
        # 앙상블 시스템
        from models.cloth_segmentation_ensemble import ClothSegmentationEnsemble
        logger.info("✅ ClothSegmentationEnsemble import 성공")
        
        # 통합 모델 관리자
        from models.cloth_segmentation_models import ClothSegmentationModels
        logger.info("✅ ClothSegmentationModels import 성공")
        
        # Attention 모델
        from models.cloth_segmentation_attention import MultiHeadSelfAttention
        logger.info("✅ MultiHeadSelfAttention import 성공")
        
        # U2Net 모델
        from models.cloth_segmentation_u2net import U2NET
        logger.info("✅ U2NET import 성공")
        
        # DeepLabV3+ 모델
        from models.cloth_segmentation_deeplabv3plus import DeepLabV3PlusModel
        logger.info("✅ DeepLabV3PlusModel import 성공")
        
        # SAM 모델
        from models.cloth_segmentation_sam import SAM
        logger.info("✅ SAM import 성공")
        
        logger.info("🎉 모든 모듈 import 성공!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        return False

def test_ensemble_system():
    """앙상블 시스템 테스트"""
    logger.info("🔄 앙상블 시스템 테스트 시작...")
    
    try:
        from models.cloth_segmentation_ensemble import ClothSegmentationEnsemble
        
        # 앙상블 시스템 생성
        ensemble = ClothSegmentationEnsemble()
        logger.info("✅ 앙상블 시스템 생성 성공")
        
        # 앙상블 정보 확인
        info = ensemble.get_ensemble_info()
        logger.info(f"✅ 앙상블 정보: {info}")
        
        # 가상의 예측 결과로 테스트
        import numpy as np
        
        # 테스트용 예측 결과 생성
        test_predictions = [
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(100, 100).astype(np.float32),
            np.random.rand(100, 100).astype(np.float32)
        ]
        
        # 가중 평균 앙상블 테스트
        result = ensemble.ensemble_predictions(test_predictions, method="weighted_average")
        if result is not None and result.shape == (100, 100):
            logger.info("✅ 가중 평균 앙상블 테스트 성공")
        else:
            logger.error("❌ 가중 평균 앙상블 테스트 실패")
            return False
            
        # 투표 앙상블 테스트
        result = ensemble.ensemble_predictions(test_predictions, method="voting")
        if result is not None and result.shape == (100, 100):
            logger.info("✅ 투표 앙상블 테스트 성공")
        else:
            logger.error("❌ 투표 앙상블 테스트 실패")
            return False
            
        logger.info("🎉 앙상블 시스템 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 앙상블 시스템 테스트 실패: {e}")
        return False

def test_model_factory():
    """모델 팩토리 테스트"""
    logger.info("🔄 모델 팩토리 테스트 시작...")
    
    try:
        from models.cloth_segmentation_models import ClothSegmentationModelFactory
        
        # 팩토리 생성
        factory = ClothSegmentationModelFactory()
        logger.info("✅ 모델 팩토리 생성 성공")
        
        # 사용 가능한 모델 목록 확인
        available_models = factory.get_available_models()
        logger.info(f"✅ 사용 가능한 모델: {available_models}")
        
        # U2Net 모델 생성 테스트
        u2net_model = factory.create_model("u2net", in_ch=3, out_ch=1)
        if u2net_model is not None:
            logger.info("✅ U2Net 모델 생성 성공")
        else:
            logger.error("❌ U2Net 모델 생성 실패")
            return False
            
        # DeepLabV3+ 모델 생성 테스트
        deeplabv3plus_model = factory.create_model("deeplabv3plus", num_classes=1)
        if deeplabv3plus_model is not None:
            logger.info("✅ DeepLabV3+ 모델 생성 성공")
        else:
            logger.error("❌ DeepLabV3+ 모델 생성 실패")
            return False
            
        logger.info("🎉 모델 팩토리 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 팩토리 테스트 실패: {e}")
        return False

def test_integrated_manager():
    """통합 모델 관리자 테스트"""
    logger.info("🔄 통합 모델 관리자 테스트 시작...")
    
    try:
        from models.cloth_segmentation_models import ClothSegmentationModels
        
        # 통합 관리자 생성
        manager = ClothSegmentationModels()
        logger.info("✅ 통합 모델 관리자 생성 성공")
        
        # 모델 정보 확인
        info = manager.get_model_info()
        logger.info(f"✅ 모델 정보: {info}")
        
        # U2Net 모델 생성
        u2net_model = manager.create_model("u2net", in_ch=3, out_ch=1)
        if u2net_model is not None:
            logger.info("✅ U2Net 모델 생성 및 등록 성공")
        else:
            logger.error("❌ U2Net 모델 생성 실패")
            return False
            
        # DeepLabV3+ 모델 생성
        deeplabv3plus_model = manager.create_model("deeplabv3plus", num_classes=1)
        if deeplabv3plus_model is not None:
            logger.info("✅ DeepLabV3+ 모델 생성 및 등록 성공")
        else:
            logger.error("❌ DeepLabV3+ 모델 생성 실패")
            return False
            
        # 등록된 모델 확인
        created_models = manager.get_all_models()
        logger.info(f"✅ 등록된 모델: {list(created_models.keys())}")
        
        # 앙상블 테스트
        if manager.ensemble_system:
            logger.info("✅ 앙상블 시스템 사용 가능")
        else:
            logger.warning("⚠️ 앙상블 시스템 사용 불가")
            
        logger.info("🎉 통합 모델 관리자 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 통합 모델 관리자 테스트 실패: {e}")
        return False

def test_attention_mechanisms():
    """Attention 메커니즘 테스트"""
    logger.info("🔄 Attention 메커니즘 테스트 시작...")
    
    try:
        from models.cloth_segmentation_attention import (
            MultiHeadSelfAttention, 
            PositionalEncoding,
            CrossAttention,
            AttentionModel
        )
        
        logger.info("✅ 모든 Attention 모듈 import 성공")
        
        # MultiHeadSelfAttention 테스트
        attention = MultiHeadSelfAttention(embed_dim=512, num_heads=8)
        logger.info("✅ MultiHeadSelfAttention 생성 성공")
        
        # PositionalEncoding 테스트
        pos_encoding = PositionalEncoding(embed_dim=512, max_seq_len=1000)
        logger.info("✅ PositionalEncoding 생성 성공")
        
        # CrossAttention 테스트
        cross_attention = CrossAttention(embed_dim=512, num_heads=8)
        logger.info("✅ CrossAttention 생성 성공")
        
        # AttentionModel 테스트
        attention_seg = AttentionModel(embed_dim=256, num_heads=8, num_layers=6)
        logger.info("✅ AttentionModel 생성 성공")
        
        logger.info("🎉 Attention 메커니즘 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Attention 메커니즘 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 의류 세그멘테이션 시스템 통합 테스트 시작!")
    
    # 테스트 결과
    test_results = []
    
    # 1. Import 테스트
    test_results.append(("Import 테스트", test_imports()))
    
    # 2. 앙상블 시스템 테스트
    test_results.append(("앙상블 시스템 테스트", test_ensemble_system()))
    
    # 3. 모델 팩토리 테스트
    test_results.append(("모델 팩토리 테스트", test_model_factory()))
    
    # 4. 통합 모델 관리자 테스트
    test_results.append(("통합 모델 관리자 테스트", test_integrated_manager()))
    
    # 5. Attention 메커니즘 테스트
    test_results.append(("Attention 메커니즘 테스트", test_attention_mechanisms()))
    
    # 테스트 결과 요약
    logger.info("\n" + "="*60)
    logger.info("📊 테스트 결과 요약")
    logger.info("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
            
    logger.info(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        return True
    else:
        logger.error(f"⚠️ {total-passed}개 테스트 실패. 문제를 확인해주세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
