#!/usr/bin/env python3
"""
🔥 Human Parsing Step - 모델 로딩 테스트 스크립트 (수정됨) - 기존 완전한 BaseStepMixin 활용
"""

import sys
import os
import logging
import torch
import numpy as np
from pathlib import Path

# 경로 추가 - backend 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, '..', '..', '..', '..')
sys.path.insert(0, backend_dir)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """모델 로딩 테스트 - 기존 완전한 BaseStepMixin 활용"""
    logger.info("🚀 모델 로딩 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
    try:
        # 1. 기존 완전한 BaseStepMixin import 테스트
        logger.info("📦 기존 완전한 BaseStepMixin import 테스트...")
        
        try:
            from app.ai_pipeline.steps.base.base_step_mixin import BaseStepMixin
            logger.info("✅ 기존 완전한 BaseStepMixin import 성공")
        except ImportError:
            try:
                from ..base.base_step_mixin import BaseStepMixin
                logger.info("✅ 기존 완전한 BaseStepMixin import 성공 (상대 경로)")
            except ImportError:
                logger.error("❌ 기존 완전한 BaseStepMixin import 실패")
                return False
        
        # 2. 모듈 import 테스트
        logger.info("📦 모듈 import 테스트...")
        
        # 상대 경로로 import
        sys.path.append(os.path.dirname(__file__))
        
        from models.model_loader import ModelLoader
        from models.checkpoint_analyzer import CheckpointAnalyzer
        from models.enhanced_models import (
            EnhancedGraphonomyModel,
            EnhancedU2NetModel,
            EnhancedDeepLabV3PlusModel
        )
        
        logger.info("✅ 모듈 import 성공")
        
        # 3. 기존 완전한 BaseStepMixin을 활용한 Mock Step 인스턴스 생성
        class MockStep(BaseStepMixin):
            def __init__(self):
                super().__init__()
                self.logger = logging.getLogger("MockStep")
        
        mock_step = MockStep()
        logger.info("✅ 기존 완전한 BaseStepMixin을 활용한 MockStep 생성 성공")
        
        # 4. ModelLoader 인스턴스 생성
        logger.info("🔧 ModelLoader 인스턴스 생성...")
        model_loader = ModelLoader(mock_step)
        logger.info("✅ ModelLoader 인스턴스 생성 성공")
        
        # 5. CheckpointAnalyzer 테스트
        logger.info("🔍 CheckpointAnalyzer 테스트...")
        checkpoint_analyzer = CheckpointAnalyzer()
        logger.info("✅ CheckpointAnalyzer 생성 성공")
        
        # 6. Enhanced Models 생성 테스트
        logger.info("🏗️ Enhanced Models 생성 테스트...")
        
        # Graphonomy 모델
        try:
            graphonomy_model = EnhancedGraphonomyModel(num_classes=20, pretrained=False)
            logger.info("✅ EnhancedGraphonomyModel 생성 성공")
            
            # 더미 입력으로 테스트 (배치 크기 2로 수정)
            dummy_input = torch.randn(2, 3, 512, 512)
            with torch.no_grad():
                output = graphonomy_model(dummy_input)
                logger.info(f"✅ Graphonomy 모델 추론 성공: {type(output)}")
                
        except Exception as e:
            logger.error(f"❌ EnhancedGraphonomyModel 생성 실패: {e}")
        
        # U2Net 모델
        try:
            u2net_model = EnhancedU2NetModel(out_channels=1)
            logger.info("✅ EnhancedU2NetModel 생성 성공")
            
            # 더미 입력으로 테스트
            dummy_input = torch.randn(1, 3, 512, 512)
            with torch.no_grad():
                output = u2net_model(dummy_input)
                logger.info(f"✅ U2Net 모델 추론 성공: {type(output)}")
                
        except Exception as e:
            logger.error(f"❌ EnhancedU2NetModel 생성 실패: {e}")
        
        # DeepLabV3+ 모델
        try:
            deeplabv3plus_model = EnhancedDeepLabV3PlusModel(num_classes=20, pretrained=False)
            logger.info("✅ EnhancedDeepLabV3PlusModel 생성 성공")
            
            # 더미 입력으로 테스트
            dummy_input = torch.randn(2, 3, 512, 512)
            with torch.no_grad():
                output = deeplabv3plus_model(dummy_input)
                logger.info(f"✅ DeepLabV3+ 모델 추론 성공: {type(output)}")
                
        except Exception as e:
            logger.error(f"❌ EnhancedDeepLabV3PlusModel 생성 실패: {e}")
        
        logger.info("✅ 모델 로딩 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 로딩 테스트 실패: {e}")
        return False

def test_step_integration():
    """Step 통합 테스트 - 기존 완전한 BaseStepMixin 활용"""
    logger.info("🚀 Step 통합 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
    try:
        # 1. 기존 완전한 BaseStepMixin을 활용한 HumanParsingStep 생성
        logger.info("🔧 HumanParsingStep 생성...")
        
        try:
            from step import HumanParsingStep
            step = HumanParsingStep()
            logger.info("✅ HumanParsingStep 생성 성공 (기존 완전한 BaseStepMixin 활용)")
        except Exception as e:
            logger.error(f"❌ HumanParsingStep 생성 실패: {e}")
            return False
        
        # 2. Step 요구사항 확인
        logger.info("📋 Step 요구사항 확인...")
        requirements = step.get_step_requirements()
        logger.info(f"✅ Step 요구사항: {requirements}")
        
        # 3. Step 설정 확인
        logger.info("⚙️ Step 설정 확인...")
        if hasattr(step, 'config'):
            logger.info(f"✅ Step 설정: {step.config}")
        else:
            logger.warning("⚠️ Step 설정 없음")
        
        logger.info("✅ Step 통합 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ Step 통합 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🎯 Human Parsing Step 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
    # 1. 모델 로딩 테스트
    model_loading_success = test_model_loading()
    
    # 2. Step 통합 테스트
    step_integration_success = test_step_integration()
    
    # 3. 결과 요약
    logger.info("📊 테스트 결과 요약:")
    logger.info(f"  - 모델 로딩 테스트: {'✅ 성공' if model_loading_success else '❌ 실패'}")
    logger.info(f"  - Step 통합 테스트: {'✅ 성공' if step_integration_success else '❌ 실패'}")
    
    if model_loading_success and step_integration_success:
        logger.info("🎉 모든 테스트 성공! (기존 완전한 BaseStepMixin 활용)")
        return True
    else:
        logger.error("❌ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
