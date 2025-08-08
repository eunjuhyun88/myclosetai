#!/usr/bin/env python3
"""
🔥 문제가 있는 AI 모델 수정 도구
================================

분석에서 발견된 문제가 있는 AI 모델들을 수정하는 도구

Author: MyCloset AI Team
Date: 2025-08-08
Version: 1.0
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

# PyTorch 관련
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# SafeTensors 관련
try:
    import safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProblematicModelFixer:
    """문제가 있는 AI 모델 수정 도구"""
    
    def __init__(self, analysis_file: str = "comprehensive_ai_model_analysis.json"):
        self.analysis_file = analysis_file
        self.analysis_data = None
        self.problematic_models = []
        self.fixed_models = []
        
    def load_analysis_data(self) -> bool:
        """분석 데이터 로드"""
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                self.analysis_data = json.load(f)
            
            # 문제가 있는 모델들 찾기
            for model_path, model_info in self.analysis_data['models'].items():
                if not model_info['valid']:
                    self.problematic_models.append({
                        'path': model_path,
                        'info': model_info
                    })
            
            print(f"🔍 발견된 문제 모델: {len(self.problematic_models)}개")
            return True
            
        except Exception as e:
            print(f"❌ 분석 데이터 로드 실패: {e}")
            return False
    
    def analyze_problematic_models(self):
        """문제가 있는 모델들 분석"""
        print("\n📋 문제가 있는 모델들 분석:")
        
        for i, model in enumerate(self.problematic_models, 1):
            path = model['path']
            info = model['info']
            
            print(f"\n{i}. {Path(path).name}")
            print(f"   📁 경로: {path}")
            print(f"   📊 크기: {info['size_mb']:.1f}MB")
            print(f"   🎯 Step: {info['step_category']}")
            print(f"   🏗️ 구조: {info['structure_type']}")
            
            if info['issues']:
                print(f"   ⚠️ 문제점:")
                for issue in info['issues']:
                    print(f"      - {issue}")
            
            if info['architecture_hints']:
                print(f"   🏛️ 아키텍처: {', '.join(info['architecture_hints'])}")
    
    def fix_header_too_large_models(self):
        """헤더가 너무 큰 모델들 수정"""
        print("\n🔧 헤더가 너무 큰 모델들 수정 시도:")
        
        for model in self.problematic_models:
            path = model['path']
            info = model['info']
            
            # 헤더가 너무 큰 문제 확인
            if any("header too large" in issue for issue in info['issues']):
                print(f"\n🔧 수정 시도: {Path(path).name}")
                
                if self._fix_header_too_large_model(path):
                    self.fixed_models.append(path)
                    print(f"✅ 수정 완료: {Path(path).name}")
                else:
                    print(f"❌ 수정 실패: {Path(path).name}")
    
    def _fix_header_too_large_model(self, model_path: str) -> bool:
        """헤더가 너무 큰 모델 수정"""
        try:
            # 백업 생성
            backup_path = f"{model_path}.backup"
            if not Path(backup_path).exists():
                shutil.copy2(model_path, backup_path)
                print(f"   📦 백업 생성: {backup_path}")
            
            # 방법 1: torch.load with map_location='cpu'
            try:
                print(f"   🔄 방법 1: torch.load with map_location='cpu' 시도")
                model_data = torch.load(model_path, map_location='cpu')
                
                # 수정된 모델 저장
                torch.save(model_data, model_path)
                print(f"   ✅ 방법 1 성공")
                return True
                
            except Exception as e1:
                print(f"   ❌ 방법 1 실패: {e1}")
                
                # 방법 2: weights_only=True
                try:
                    print(f"   🔄 방법 2: weights_only=True 시도")
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    
                    # 수정된 모델 저장
                    torch.save(model_data, model_path)
                    print(f"   ✅ 방법 2 성공")
                    return True
                    
                except Exception as e2:
                    print(f"   ❌ 방법 2 실패: {e2}")
                    
                    # 방법 3: SafeTensors로 변환
                    if SAFETENSORS_AVAILABLE:
                        try:
                            print(f"   🔄 방법 3: SafeTensors 변환 시도")
                            if self._convert_to_safetensors(model_path):
                                print(f"   ✅ 방법 3 성공")
                                return True
                        except Exception as e3:
                            print(f"   ❌ 방법 3 실패: {e3}")
                    
                    # 방법 4: 파일 재구성
                    try:
                        print(f"   🔄 방법 4: 파일 재구성 시도")
                        if self._reconstruct_model_file(model_path):
                            print(f"   ✅ 방법 4 성공")
                            return True
                    except Exception as e4:
                        print(f"   ❌ 방법 4 실패: {e4}")
            
            return False
            
        except Exception as e:
            print(f"   ❌ 수정 중 오류: {e}")
            return False
    
    def _convert_to_safetensors(self, model_path: str) -> bool:
        """PyTorch 모델을 SafeTensors로 변환"""
        try:
            # 임시로 모델 로딩 시도
            model_data = None
            
            # 다양한 로딩 방법 시도
            for method in ['weights_only_true', 'weights_only_false']:
                try:
                    if method == 'weights_only_true':
                        model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    elif method == 'weights_only_false':
                        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    
                    if model_data is not None:
                        break
                except:
                    continue
            
            if model_data is None:
                return False
            
            # SafeTensors로 저장
            safetensors_path = model_path.replace('.pth', '.safetensors').replace('.pt', '.safetensors')
            
            if isinstance(model_data, dict):
                # state_dict 형태인 경우
                if 'state_dict' in model_data:
                    safetensors.save_file(model_data['state_dict'], safetensors_path)
                else:
                    safetensors.save_file(model_data, safetensors_path)
            else:
                # 직접 텐서인 경우
                safetensors.save_file({'model': model_data}, safetensors_path)
            
            # 원본 파일 백업하고 SafeTensors 파일로 교체
            shutil.move(model_path, f"{model_path}.old")
            shutil.move(safetensors_path, model_path)
            
            return True
            
        except Exception as e:
            print(f"   ❌ SafeTensors 변환 실패: {e}")
            return False
    
    def _reconstruct_model_file(self, model_path: str) -> bool:
        """모델 파일 재구성"""
        try:
            # 파일 크기 확인
            file_size = Path(model_path).stat().st_size
            
            if file_size == 0:
                print(f"   ⚠️ 파일 크기가 0입니다")
                return False
            
            # 파일의 처음 부분 읽기
            with open(model_path, 'rb') as f:
                header = f.read(1024)  # 처음 1KB 읽기
            
            # PyTorch 시그니처 확인
            if b'PK\x03\x04' in header:  # ZIP 파일 시그니처
                print(f"   🔍 ZIP 파일로 감지됨")
                return self._fix_zip_model_file(model_path)
            elif b'pytorch' in header.lower():
                print(f"   🔍 PyTorch 파일로 감지됨")
                return self._fix_pytorch_model_file(model_path)
            else:
                print(f"   🔍 알 수 없는 파일 형식")
                return False
                
        except Exception as e:
            print(f"   ❌ 파일 재구성 실패: {e}")
            return False
    
    def _fix_zip_model_file(self, model_path: str) -> bool:
        """ZIP 형태의 모델 파일 수정"""
        try:
            import zipfile
            
            # ZIP 파일로 열기
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                # 압축 해제
                temp_dir = f"{model_path}_temp"
                zip_ref.extractall(temp_dir)
                
                # data.pkl 파일 찾기
                data_pkl_path = Path(temp_dir) / "data.pkl"
                if data_pkl_path.exists():
                    # data.pkl을 다시 로드하고 저장
                    model_data = torch.load(str(data_pkl_path), map_location='cpu')
                    torch.save(model_data, model_path)
                    
                    # 임시 디렉토리 정리
                    shutil.rmtree(temp_dir)
                    return True
            
            return False
            
        except Exception as e:
            print(f"   ❌ ZIP 파일 수정 실패: {e}")
            return False
    
    def _fix_pytorch_model_file(self, model_path: str) -> bool:
        """PyTorch 모델 파일 수정"""
        try:
            # 파일을 바이너리로 읽기
            with open(model_path, 'rb') as f:
                data = f.read()
            
            # PyTorch 헤더 찾기
            pytorch_header = b'pytorch'
            header_pos = data.find(pytorch_header)
            
            if header_pos != -1:
                # 헤더 부분을 건너뛰고 데이터 부분만 추출
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                torch.save(model_data, model_path)
                return True
            
            return False
            
        except Exception as e:
            print(f"   ❌ PyTorch 파일 수정 실패: {e}")
            return False
    
    def fix_invalid_json_models(self):
        """잘못된 JSON 헤더 모델들 수정"""
        print("\n🔧 잘못된 JSON 헤더 모델들 수정 시도:")
        
        for model in self.problematic_models:
            path = model['path']
            info = model['info']
            
            # 잘못된 JSON 문제 확인
            if any("invalid JSON" in issue for issue in info['issues']):
                print(f"\n🔧 수정 시도: {Path(path).name}")
                
                if self._fix_invalid_json_model(path):
                    self.fixed_models.append(path)
                    print(f"✅ 수정 완료: {Path(path).name}")
                else:
                    print(f"❌ 수정 실패: {Path(path).name}")
    
    def _fix_invalid_json_model(self, model_path: str) -> bool:
        """잘못된 JSON 헤더 모델 수정"""
        try:
            # 백업 생성
            backup_path = f"{model_path}.backup"
            if not Path(backup_path).exists():
                shutil.copy2(model_path, backup_path)
                print(f"   📦 백업 생성: {backup_path}")
            
            # 파일을 바이너리로 읽기
            with open(model_path, 'rb') as f:
                data = f.read()
            
            # PyTorch 시그니처 찾기
            pytorch_signature = b'PK\x03\x04'  # ZIP 파일 시그니처
            
            if pytorch_signature in data:
                # ZIP 파일로 처리
                return self._fix_zip_model_file(model_path)
            else:
                # 다른 방법 시도
                try:
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    torch.save(model_data, model_path)
                    return True
                except:
                    return False
            
        except Exception as e:
            print(f"   ❌ JSON 헤더 수정 실패: {e}")
            return False
    
    def fix_missing_file_models(self):
        """파일이 없는 모델들 처리"""
        print("\n🔧 파일이 없는 모델들 처리:")
        
        for model in self.problematic_models:
            path = model['path']
            info = model['info']
            
            # 파일이 없는 문제 확인
            if any("파일이 존재하지 않음" in issue for issue in info['issues']):
                print(f"\n⚠️ 파일 없음: {Path(path).name}")
                print(f"   📁 경로: {path}")
                
                # 대체 파일 찾기
                alternative_file = self._find_alternative_file(path)
                if alternative_file:
                    print(f"   🔍 대체 파일 발견: {Path(alternative_file).name}")
                    
                    # 심볼릭 링크 생성
                    try:
                        if Path(path).parent.exists():
                            os.symlink(alternative_file, path)
                            print(f"   ✅ 심볼릭 링크 생성 완료")
                            self.fixed_models.append(path)
                        else:
                            print(f"   ❌ 대상 디렉토리가 존재하지 않음")
                    except Exception as e:
                        print(f"   ❌ 심볼릭 링크 생성 실패: {e}")
                else:
                    print(f"   ❌ 대체 파일을 찾을 수 없음")
    
    def _find_alternative_file(self, original_path: str) -> Optional[str]:
        """대체 파일 찾기"""
        original_name = Path(original_path).name
        original_dir = Path(original_path).parent
        
        # 같은 디렉토리에서 비슷한 이름의 파일 찾기
        if original_dir.exists():
            for file_path in original_dir.glob("*"):
                if file_path.is_file() and file_path.name != original_name:
                    # 파일 이름이 비슷한지 확인
                    if any(keyword in file_path.name.lower() for keyword in 
                           ['model', 'pytorch', 'checkpoint', 'weights']):
                        return str(file_path)
        
        # 상위 디렉토리에서 찾기
        parent_dir = original_dir.parent
        if parent_dir.exists():
            for file_path in parent_dir.rglob("*"):
                if file_path.is_file() and file_path.name == original_name:
                    return str(file_path)
        
        return None
    
    def verify_fixed_models(self):
        """수정된 모델들 검증"""
        print(f"\n🔍 수정된 모델들 검증:")
        
        verified_count = 0
        for model_path in self.fixed_models:
            print(f"\n🔍 검증 중: {Path(model_path).name}")
            
            try:
                # 모델 로딩 테스트
                if model_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
                    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                    print(f"   ✅ SafeTensors 로딩 성공 (키 수: {len(keys)})")
                    verified_count += 1
                else:
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    print(f"   ✅ PyTorch 로딩 성공")
                    verified_count += 1
                    
            except Exception as e:
                print(f"   ❌ 검증 실패: {e}")
        
        print(f"\n📊 검증 결과: {verified_count}/{len(self.fixed_models)}개 성공")
    
    def generate_fix_report(self):
        """수정 리포트 생성"""
        report = []
        report.append("🔥 AI 모델 수정 리포트")
        report.append("=" * 80)
        report.append(f"📅 수정 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append(f"📊 전체 문제 모델: {len(self.problematic_models)}개")
        report.append(f"✅ 수정 완료: {len(self.fixed_models)}개")
        report.append(f"❌ 수정 실패: {len(self.problematic_models) - len(self.fixed_models)}개")
        report.append("")
        
        if self.fixed_models:
            report.append("✅ 수정 완료된 모델들:")
            for model_path in self.fixed_models:
                report.append(f"   - {Path(model_path).name}")
            report.append("")
        
        remaining_problems = [m for m in self.problematic_models if m['path'] not in self.fixed_models]
        if remaining_problems:
            report.append("❌ 수정 실패한 모델들:")
            for model in remaining_problems:
                report.append(f"   - {Path(model['path']).name}")
                for issue in model['info']['issues']:
                    report.append(f"     문제: {issue}")
            report.append("")
        
        return "\n".join(report)

def main():
    """메인 함수"""
    print("🔥 문제가 있는 AI 모델 수정 도구")
    print("=" * 80)
    
    # 수정기 초기화
    fixer = ProblematicModelFixer()
    
    # 1. 분석 데이터 로드
    print("\n📋 1단계: 분석 데이터 로드")
    if not fixer.load_analysis_data():
        print("❌ 분석 데이터 로드를 실패했습니다.")
        return
    
    # 2. 문제가 있는 모델들 분석
    print("\n📋 2단계: 문제가 있는 모델들 분석")
    fixer.analyze_problematic_models()
    
    # 3. 헤더가 너무 큰 모델들 수정
    print("\n🔧 3단계: 헤더가 너무 큰 모델들 수정")
    fixer.fix_header_too_large_models()
    
    # 4. 잘못된 JSON 헤더 모델들 수정
    print("\n🔧 4단계: 잘못된 JSON 헤더 모델들 수정")
    fixer.fix_invalid_json_models()
    
    # 5. 파일이 없는 모델들 처리
    print("\n🔧 5단계: 파일이 없는 모델들 처리")
    fixer.fix_missing_file_models()
    
    # 6. 수정된 모델들 검증
    print("\n🔍 6단계: 수정된 모델들 검증")
    fixer.verify_fixed_models()
    
    # 7. 수정 리포트 생성
    print("\n📋 7단계: 수정 리포트 생성")
    report = fixer.generate_fix_report()
    print(report)
    
    # 8. 리포트 저장
    with open("ai_model_fix_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n💾 수정 리포트 저장: ai_model_fix_report.txt")
    print("\n🎉 AI 모델 수정 완료!")

if __name__ == "__main__":
    from datetime import datetime
    main()
