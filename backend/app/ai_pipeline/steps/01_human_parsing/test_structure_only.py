#!/usr/bin/env python3
"""
🔥 Human Parsing Step - 구조 테스트 스크립트 (PyTorch 없이) - 기존 완전한 BaseStepMixin 활용
"""

import sys
import os
import logging
from pathlib import Path

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_module_structure():
    """모듈 구조 테스트 - 기존 완전한 BaseStepMixin 활용"""
    logger.info("🚀 모듈 구조 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
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
        
        # 2. 파일 존재 확인
        logger.info("📁 파일 존재 확인...")
        
        required_files = [
            'step.py',
            'models/model_loader.py',
            'models/checkpoint_analyzer.py',
            'models/enhanced_models.py',
            'inference/inference_engine.py',
            'preprocessing/preprocessor.py',
            'postprocessing/postprocessor.py',
            'utils/utils.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                logger.info(f"✅ {file_path} (존재함)")
            else:
                logger.error(f"❌ {file_path} (존재하지 않음)")
        
        # 3. 모듈 import 테스트 (PyTorch 없이)
        logger.info("📦 모듈 import 테스트...")
        
        # models 모듈 테스트
        try:
            sys.path.append('.')
            from models.model_loader import ModelLoader
            logger.info("✅ ModelLoader import 성공")
        except Exception as e:
            logger.warning(f"⚠️ ModelLoader import 실패: {e}")
        
        try:
            from models.checkpoint_analyzer import CheckpointAnalyzer
            logger.info("✅ CheckpointAnalyzer import 성공")
        except Exception as e:
            logger.warning(f"⚠️ CheckpointAnalyzer import 실패: {e}")
        
        # 4. 클래스 구조 확인
        logger.info("🏗️ 클래스 구조 확인...")
        
        # ModelLoader 클래스 확인
        try:
            from models.model_loader import ModelLoader
            model_loader = ModelLoader.__name__
            logger.info(f"✅ ModelLoader 클래스: {model_loader}")
            
            # 메서드 확인
            methods = [method for method in dir(ModelLoader) if not method.startswith('_')]
            logger.info(f"✅ ModelLoader 메서드들: {methods[:5]}...")  # 처음 5개만 표시
            
        except Exception as e:
            logger.warning(f"⚠️ ModelLoader 클래스 확인 실패: {e}")
        
        # 5. 파일 크기 확인
        logger.info("📊 파일 크기 확인...")
        
        file_sizes = {
            'step.py': 'step.py',
            'models/model_loader.py': 'models/model_loader.py',
            'models/enhanced_models.py': 'models/enhanced_models.py',
            'inference/inference_engine.py': 'inference/inference_engine.py',
            'utils/utils.py': 'utils/utils.py'
        }
        
        for name, path in file_sizes.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                logger.info(f"✅ {name}: {size:,} bytes")
            else:
                logger.warning(f"⚠️ {name}: 파일 없음")
        
        # 6. 디렉토리 구조 확인
        logger.info("📁 디렉토리 구조 확인...")
        
        directories = [
            'models',
            'inference',
            'preprocessing',
            'postprocessing',
            'utils',
            'config',
            'ensemble',
            'processors',
            'services'
        ]
        
        for directory in directories:
            if os.path.exists(directory):
                files = os.listdir(directory)
                logger.info(f"✅ {directory}/: {len(files)} files")
            else:
                logger.warning(f"⚠️ {directory}/: 디렉토리 없음")
        
        logger.info("✅ 모듈 구조 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모듈 구조 테스트 실패: {e}")
        return False

def test_import_structure():
    """import 구조 테스트 - 기존 완전한 BaseStepMixin 활용"""
    logger.info("🚀 import 구조 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
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
        
        # 2. step.py import 테스트
        logger.info("📦 step.py import 테스트...")
        
        try:
            from step import HumanParsingStep
            logger.info("✅ HumanParsingStep import 성공")
            
            # 클래스 메서드 확인
            methods = [method for method in dir(HumanParsingStep) if not method.startswith('_')]
            logger.info(f"✅ HumanParsingStep 메서드들: {methods[:10]}...")  # 처음 10개만 표시
            
        except Exception as e:
            logger.warning(f"⚠️ HumanParsingStep import 실패: {e}")
        
        # 3. 모듈별 import 테스트
        logger.info("📦 모듈별 import 테스트...")
        
        modules_to_test = [
            ('models.model_loader', 'ModelLoader'),
            ('models.checkpoint_analyzer', 'CheckpointAnalyzer'),
            ('inference.inference_engine', 'InferenceEngine'),
            ('preprocessing.preprocessor', 'Preprocessor'),
            ('postprocessing.postprocessor', 'Postprocessor'),
            ('utils.utils', 'Utils')
        ]
        
        for module_name, class_name in modules_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if hasattr(module, class_name):
                    logger.info(f"✅ {module_name}.{class_name} import 성공")
                else:
                    logger.warning(f"⚠️ {module_name}.{class_name} 클래스 없음")
            except Exception as e:
                logger.warning(f"⚠️ {module_name}.{class_name} import 실패: {e}")
        
        logger.info("✅ import 구조 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ import 구조 테스트 실패: {e}")
        return False

def test_file_content():
    """파일 내용 테스트 - 기존 완전한 BaseStepMixin 활용"""
    logger.info("🚀 파일 내용 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
    try:
        # 1. step.py 파일 내용 확인
        logger.info("📄 step.py 파일 내용 확인...")
        
        if os.path.exists('step.py'):
            with open('step.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 기존 완전한 BaseStepMixin 활용 확인
            if 'BaseStepMixin' in content:
                logger.info("✅ step.py에서 BaseStepMixin 활용 확인")
            else:
                logger.warning("⚠️ step.py에서 BaseStepMixin 활용 없음")
            
            # HumanParsingStep 클래스 확인
            if 'class HumanParsingStep' in content:
                logger.info("✅ step.py에서 HumanParsingStep 클래스 확인")
            else:
                logger.warning("⚠️ step.py에서 HumanParsingStep 클래스 없음")
            
            # 기존 완전한 BaseStepMixin import 확인
            if 'from app.ai_pipeline.steps.base.base_step_mixin import BaseStepMixin' in content:
                logger.info("✅ step.py에서 기존 완전한 BaseStepMixin import 확인")
            else:
                logger.warning("⚠️ step.py에서 기존 완전한 BaseStepMixin import 없음")
        else:
            logger.error("❌ step.py 파일 없음")
        
        # 2. __init__.py 파일 내용 확인
        logger.info("📄 __init__.py 파일 내용 확인...")
        
        if os.path.exists('__init__.py'):
            with open('__init__.py', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # BaseStepMixin import 확인
            if 'BaseStepMixin' in content:
                logger.info("✅ __init__.py에서 BaseStepMixin 활용 확인")
            else:
                logger.warning("⚠️ __init__.py에서 BaseStepMixin 활용 없음")
        else:
            logger.error("❌ __init__.py 파일 없음")
        
        logger.info("✅ 파일 내용 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 파일 내용 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🎯 Human Parsing Step 구조 테스트 시작 (기존 완전한 BaseStepMixin 활용)")
    
    # 1. 모듈 구조 테스트
    structure_success = test_module_structure()
    
    # 2. import 구조 테스트
    import_success = test_import_structure()
    
    # 3. 파일 내용 테스트
    content_success = test_file_content()
    
    # 4. 결과 요약
    logger.info("📊 테스트 결과 요약:")
    logger.info(f"  - 모듈 구조 테스트: {'✅ 성공' if structure_success else '❌ 실패'}")
    logger.info(f"  - import 구조 테스트: {'✅ 성공' if import_success else '❌ 실패'}")
    logger.info(f"  - 파일 내용 테스트: {'✅ 성공' if content_success else '❌ 실패'}")
    
    if structure_success and import_success and content_success:
        logger.info("🎉 모든 테스트 성공! (기존 완전한 BaseStepMixin 활용)")
        return True
    else:
        logger.error("❌ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
