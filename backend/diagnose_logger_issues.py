#!/usr/bin/env python3
"""
🔥 Step 파일별 Logger 문제 정확한 진단 스크립트
================================================================

목적: 각 Step 파일을 개별적으로 import해서 어디서 logger 에러가 나는지 정확히 찾기

실행: python diagnose_logger_issues.py
"""

import os
import sys
import traceback
import importlib
import logging
from pathlib import Path

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StepLoggerDiagnoser:
    """Step별 Logger 문제 진단기"""
    
    def __init__(self, backend_path: str = "backend"):
        self.backend_path = Path(backend_path)
        self.step_files = [
            "app.ai_pipeline.steps.step_01_human_parsing",
            "app.ai_pipeline.steps.step_02_pose_estimation", 
            "app.ai_pipeline.steps.step_03_cloth_segmentation",
            "app.ai_pipeline.steps.step_04_geometric_matching",
            "app.ai_pipeline.steps.step_05_cloth_warping",
            "app.ai_pipeline.steps.step_06_virtual_fitting",
            "app.ai_pipeline.steps.step_07_post_processing",
            "app.ai_pipeline.steps.step_08_quality_assessment"
        ]
        
    def diagnose_all_steps(self):
        """모든 Step 파일 진단"""
        print("🔥 Step별 Logger 문제 정확한 진단")
        print("=" * 60)
        
        # Python path 설정
        if str(self.backend_path) not in sys.path:
            sys.path.insert(0, str(self.backend_path))
        
        for step_module in self.step_files:
            print(f"\n🔍 진단 중: {step_module}")
            self.diagnose_single_step(step_module)
    
    def diagnose_single_step(self, step_module: str):
        """개별 Step 진단"""
        try:
            # 모듈 import 시도
            module = importlib.import_module(step_module)
            print(f"✅ {step_module} import 성공")
            
            # Step 클래스 찾기
            step_class_name = self.extract_step_class_name(step_module)
            if hasattr(module, step_class_name):
                step_class = getattr(module, step_class_name)
                print(f"✅ {step_class_name} 클래스 발견")
                
                # 클래스 인스턴스 생성 시도
                try:
                    instance = step_class()
                    print(f"✅ {step_class_name} 인스턴스 생성 성공")
                except Exception as e:
                    print(f"⚠️ {step_class_name} 인스턴스 생성 실패: {e}")
            else:
                print(f"⚠️ {step_class_name} 클래스 없음")
                
        except ImportError as e:
            error_msg = str(e)
            print(f"❌ {step_module} ImportError: {error_msg}")
            
            # logger 관련 에러인지 확인
            if 'logger' in error_msg.lower():
                print(f"🎯 LOGGER 문제 발견!")
                self.analyze_logger_error(step_module, error_msg)
            elif 'logging' in error_msg.lower():
                print(f"🎯 LOGGING 문제 발견!")
                self.analyze_logging_error(step_module, error_msg)
            else:
                print(f"📋 기타 import 문제: {error_msg}")
                
        except Exception as e:
            print(f"❌ {step_module} 예외: {e}")
            print(f"스택 트레이스:")
            traceback.print_exc()
    
    def extract_step_class_name(self, step_module: str) -> str:
        """모듈명에서 클래스명 추출"""
        step_mapping = {
            "step_01_human_parsing": "HumanParsingStep",
            "step_02_pose_estimation": "PoseEstimationStep",
            "step_03_cloth_segmentation": "ClothSegmentationStep", 
            "step_04_geometric_matching": "GeometricMatchingStep",
            "step_05_cloth_warping": "ClothWarpingStep",
            "step_06_virtual_fitting": "VirtualFittingStep",
            "step_07_post_processing": "PostProcessingStep",
            "step_08_quality_assessment": "QualityAssessmentStep"
        }
        
        for step_name, class_name in step_mapping.items():
            if step_name in step_module:
                return class_name
        
        return "Unknown"
    
    def analyze_logger_error(self, step_module: str, error_msg: str):
        """Logger 에러 상세 분석"""
        print(f"🔍 Logger 에러 상세 분석:")
        
        if "name 'logger' is not defined" in error_msg:
            print(f"   원인: logger 변수가 정의되지 않음")
            print(f"   해결: 파일 시작 부분에 'logger = logging.getLogger(__name__)' 추가")
            
        elif "logger" in error_msg:
            print(f"   원인: logger 관련 기타 문제")
            print(f"   확인 필요: logger 사용 위치 및 정의 순서")
        
        # 실제 파일 내용 확인 제안
        step_file = self.get_step_file_path(step_module)
        if step_file and step_file.exists():
            print(f"   파일 위치: {step_file}")
            self.check_logger_definition_in_file(step_file)
    
    def analyze_logging_error(self, step_module: str, error_msg: str):
        """Logging 에러 상세 분석"""
        print(f"🔍 Logging 에러 상세 분석:")
        
        if "name 'logging' is not defined" in error_msg:
            print(f"   원인: logging 모듈이 import되지 않음")
            print(f"   해결: 파일 시작 부분에 'import logging' 추가")
            
        # 실제 파일 내용 확인
        step_file = self.get_step_file_path(step_module)
        if step_file and step_file.exists():
            print(f"   파일 위치: {step_file}")
            self.check_logging_import_in_file(step_file)
    
    def get_step_file_path(self, step_module: str) -> Path:
        """Step 모듈의 실제 파일 경로 반환"""
        module_path = step_module.replace('.', '/')
        return self.backend_path / f"{module_path}.py"
    
    def check_logger_definition_in_file(self, file_path: Path):
        """파일에서 logger 정의 확인"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # logger 정의 찾기
            logger_definitions = []
            for i, line in enumerate(lines):
                if 'logger = ' in line:
                    logger_definitions.append((i+1, line.strip()))
            
            if logger_definitions:
                print(f"   📋 발견된 logger 정의:")
                for line_num, line in logger_definitions:
                    print(f"      라인 {line_num}: {line}")
            else:
                print(f"   ❌ logger 정의 없음!")
                
            # import logging 확인
            has_logging_import = any('import logging' in line for line in lines)
            print(f"   📋 'import logging' 존재: {'✅' if has_logging_import else '❌'}")
            
        except Exception as e:
            print(f"   ❌ 파일 읽기 실패: {e}")
    
    def check_logging_import_in_file(self, file_path: Path):
        """파일에서 logging import 확인"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # logging import 찾기
            logging_imports = []
            for i, line in enumerate(lines):
                if 'import logging' in line or 'from logging' in line:
                    logging_imports.append((i+1, line.strip()))
            
            if logging_imports:
                print(f"   📋 발견된 logging import:")
                for line_num, line in logging_imports:
                    print(f"      라인 {line_num}: {line}")
            else:
                print(f"   ❌ logging import 없음!")
                
        except Exception as e:
            print(f"   ❌ 파일 읽기 실패: {e}")

def main():
    """메인 실행"""
    diagnoser = StepLoggerDiagnoser()
    diagnoser.diagnose_all_steps()
    
    print("\n" + "=" * 60)
    print("🎯 진단 완료!")
    print("📝 다음 단계:")
    print("   1. 각 Step 파일에서 발견된 문제들을 수정")
    print("   2. logger = logging.getLogger(__name__) 확인")
    print("   3. import logging 확인")
    print("   4. logger 사용 위치와 정의 순서 확인")

if __name__ == "__main__":
    main() 