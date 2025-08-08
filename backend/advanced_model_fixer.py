#!/usr/bin/env python3
"""
🔥 고급 AI 모델 수정 도구
========================

문제가 있는 AI 모델들을 고급 기술로 수정하는 도구

Author: MyCloset AI Team
Date: 2025-08-08
Version: 2.0
"""

import os
import sys
import json
import shutil
import logging
import zipfile
import pickle
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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

class AdvancedModelFixer:
    """고급 AI 모델 수정 도구"""
    
    def __init__(self):
        self.fixed_models = []
        self.failed_models = []
        
    def fix_all_problematic_models(self):
        """모든 문제가 있는 모델들 수정"""
        print("🔥 고급 AI 모델 수정 도구")
        print("=" * 80)
        
        # 문제가 있는 모델들 목록
        problematic_models = [
            "backend/ai_models/Graphonomy/training_args.bin",
            "backend/ai_models/step_01_human_parsing/graphonomy.pth",
            "backend/ai_models/step_01_human_parsing/graphonomy_root.pth",
            "backend/ai_models/step_03_cloth_segmentation/u2net_official.pth",
            "backend/ai_models/step_06_virtual_fitting/ootdiffusion/text_encoder/ootdiffusion/text_encoder/text_encoder_pytorch_model.bin",
            "backend/ai_models/step_06_virtual_fitting/ootdiffusion/vae/ootdiffusion/vae/vae_diffusion_pytorch_model.bin"
        ]
        
        for model_path in problematic_models:
            if Path(model_path).exists():
                print(f"\n🔧 수정 시도: {Path(model_path).name}")
                if self.fix_model(model_path):
                    self.fixed_models.append(model_path)
                    print(f"✅ 수정 완료: {Path(model_path).name}")
                else:
                    self.failed_models.append(model_path)
                    print(f"❌ 수정 실패: {Path(model_path).name}")
            else:
                print(f"\n⚠️ 파일 없음: {model_path}")
        
        # 결과 출력
        print(f"\n📊 수정 결과:")
        print(f"   ✅ 성공: {len(self.fixed_models)}개")
        print(f"   ❌ 실패: {len(self.failed_models)}개")
    
    def fix_model(self, model_path: str) -> bool:
        """개별 모델 수정"""
        try:
            # 백업 생성
            backup_path = f"{model_path}.backup"
            if not Path(backup_path).exists():
                shutil.copy2(model_path, backup_path)
                print(f"   📦 백업 생성: {backup_path}")
            
            # 파일 크기 확인
            file_size = Path(model_path).stat().st_size
            if file_size == 0:
                print(f"   ⚠️ 파일 크기가 0입니다")
                return False
            
            # 파일 타입 감지
            file_type = self._detect_file_type(model_path)
            print(f"   🔍 파일 타입: {file_type}")
            
            # 파일 타입별 수정
            if file_type == "zip":
                return self._fix_zip_model(model_path)
            elif file_type == "pytorch":
                return self._fix_pytorch_model(model_path)
            elif file_type == "corrupted":
                return self._fix_corrupted_model(model_path)
            else:
                return self._fix_unknown_model(model_path)
                
        except Exception as e:
            print(f"   ❌ 수정 중 오류: {e}")
            return False
    
    def _detect_file_type(self, model_path: str) -> str:
        """파일 타입 감지"""
        try:
            with open(model_path, 'rb') as f:
                header = f.read(1024)
            
            # ZIP 파일 시그니처
            if header.startswith(b'PK\x03\x04'):
                return "zip"
            
            # PyTorch 시그니처
            if b'pytorch' in header.lower():
                return "pytorch"
            
            # 손상된 파일 감지
            if len(header) < 100 or all(b == 0 for b in header[:100]):
                return "corrupted"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def _fix_zip_model(self, model_path: str) -> bool:
        """ZIP 형태의 모델 수정"""
        try:
            print(f"   🔄 ZIP 파일 수정 시도")
            
            # 임시 디렉토리 생성
            temp_dir = f"{model_path}_temp"
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            
            # ZIP 파일 압축 해제
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # data.pkl 파일 찾기
            data_pkl_path = Path(temp_dir) / "data.pkl"
            if data_pkl_path.exists():
                print(f"   🔍 data.pkl 발견")
                
                # data.pkl 로딩 시도
                try:
                    model_data = torch.load(str(data_pkl_path), map_location='cpu')
                    torch.save(model_data, model_path)
                    print(f"   ✅ data.pkl에서 모델 데이터 추출 성공")
                    
                    # 임시 디렉토리 정리
                    shutil.rmtree(temp_dir)
                    return True
                    
                except Exception as e:
                    print(f"   ❌ data.pkl 로딩 실패: {e}")
            
            # 다른 파일들 확인
            for file_path in Path(temp_dir).rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.pkl', '.pt', '.pth']:
                    try:
                        model_data = torch.load(str(file_path), map_location='cpu')
                        torch.save(model_data, model_path)
                        print(f"   ✅ {file_path.name}에서 모델 데이터 추출 성공")
                        
                        # 임시 디렉토리 정리
                        shutil.rmtree(temp_dir)
                        return True
                        
                    except Exception as e:
                        print(f"   ❌ {file_path.name} 로딩 실패: {e}")
                        continue
            
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir)
            return False
            
        except Exception as e:
            print(f"   ❌ ZIP 파일 수정 실패: {e}")
            return False
    
    def _fix_pytorch_model(self, model_path: str) -> bool:
        """PyTorch 모델 수정"""
        try:
            print(f"   🔄 PyTorch 모델 수정 시도")
            
            # 방법 1: weights_only=False (보안 주의)
            try:
                print(f"   🔄 방법 1: weights_only=False 시도")
                model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                torch.save(model_data, model_path)
                print(f"   ✅ 방법 1 성공")
                return True
            except Exception as e1:
                print(f"   ❌ 방법 1 실패: {e1}")
            
            # 방법 2: SafeTensors 변환
            if SAFETENSORS_AVAILABLE:
                try:
                    print(f"   🔄 방법 2: SafeTensors 변환 시도")
                    if self._convert_to_safetensors(model_path):
                        print(f"   ✅ 방법 2 성공")
                        return True
                except Exception as e2:
                    print(f"   ❌ 방법 2 실패: {e2}")
            
            # 방법 3: 파일 재구성
            try:
                print(f"   🔄 방법 3: 파일 재구성 시도")
                if self._reconstruct_pytorch_file(model_path):
                    print(f"   ✅ 방법 3 성공")
                    return True
            except Exception as e3:
                print(f"   ❌ 방법 3 실패: {e3}")
            
            return False
            
        except Exception as e:
            print(f"   ❌ PyTorch 모델 수정 실패: {e}")
            return False
    
    def _fix_corrupted_model(self, model_path: str) -> bool:
        """손상된 모델 수정"""
        try:
            print(f"   🔄 손상된 모델 수정 시도")
            
            # 파일 크기 확인
            file_size = Path(model_path).stat().st_size
            
            if file_size == 0:
                print(f"   ⚠️ 파일이 비어있습니다")
                return False
            
            # 파일의 처음 부분 읽기
            with open(model_path, 'rb') as f:
                data = f.read()
            
            # PyTorch 시그니처 찾기
            pytorch_signatures = [b'PK\x03\x04', b'pytorch', b'pickle']
            
            for signature in pytorch_signatures:
                pos = data.find(signature)
                if pos != -1:
                    print(f"   🔍 PyTorch 시그니처 발견: {signature}")
                    
                    # 시그니처부터 끝까지 추출
                    valid_data = data[pos:]
                    
                    # 임시 파일에 저장
                    temp_path = f"{model_path}_temp"
                    with open(temp_path, 'wb') as f:
                        f.write(valid_data)
                    
                    # 임시 파일 테스트
                    try:
                        model_data = torch.load(temp_path, map_location='cpu', weights_only=True)
                        torch.save(model_data, model_path)
                        os.remove(temp_path)
                        print(f"   ✅ 손상된 파일에서 유효한 데이터 추출 성공")
                        return True
                    except Exception as e:
                        print(f"   ❌ 임시 파일 로딩 실패: {e}")
                        os.remove(temp_path)
            
            return False
            
        except Exception as e:
            print(f"   ❌ 손상된 모델 수정 실패: {e}")
            return False
    
    def _fix_unknown_model(self, model_path: str) -> bool:
        """알 수 없는 모델 수정"""
        try:
            print(f"   🔄 알 수 없는 모델 수정 시도")
            
            # 다양한 방법 시도
            methods = [
                self._try_weights_only_false,
                self._try_safetensors_conversion,
                self._try_file_reconstruction,
                self._try_binary_repair
            ]
            
            for i, method in enumerate(methods, 1):
                try:
                    print(f"   🔄 방법 {i} 시도")
                    if method(model_path):
                        print(f"   ✅ 방법 {i} 성공")
                        return True
                except Exception as e:
                    print(f"   ❌ 방법 {i} 실패: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"   ❌ 알 수 없는 모델 수정 실패: {e}")
            return False
    
    def _try_weights_only_false(self, model_path: str) -> bool:
        """weights_only=False로 시도"""
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        torch.save(model_data, model_path)
        return True
    
    def _try_safetensors_conversion(self, model_path: str) -> bool:
        """SafeTensors 변환 시도"""
        if not SAFETENSORS_AVAILABLE:
            return False
        
        # 모델 로딩
        model_data = None
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
            if 'state_dict' in model_data:
                safetensors.save_file(model_data['state_dict'], safetensors_path)
            else:
                safetensors.save_file(model_data, safetensors_path)
        else:
            safetensors.save_file({'model': model_data}, safetensors_path)
        
        # 원본 파일 교체
        shutil.move(model_path, f"{model_path}.old")
        shutil.move(safetensors_path, model_path)
        
        return True
    
    def _try_file_reconstruction(self, model_path: str) -> bool:
        """파일 재구성 시도"""
        # 파일을 바이너리로 읽기
        with open(model_path, 'rb') as f:
            data = f.read()
        
        # PyTorch 관련 시그니처 찾기
        signatures = [
            (b'PK\x03\x04', 'zip'),
            (b'pytorch', 'pytorch'),
            (b'pickle', 'pickle'),
            (b'\x80\x02', 'pickle'),  # pickle 프로토콜 2
            (b'\x80\x03', 'pickle'),  # pickle 프로토콜 3
            (b'\x80\x04', 'pickle'),  # pickle 프로토콜 4
        ]
        
        for signature, file_type in signatures:
            pos = data.find(signature)
            if pos != -1:
                print(f"   🔍 {file_type} 시그니처 발견")
                
                # 시그니처부터 끝까지 추출
                valid_data = data[pos:]
                
                # 임시 파일에 저장
                temp_path = f"{model_path}_temp"
                with open(temp_path, 'wb') as f:
                    f.write(valid_data)
                
                # 임시 파일 테스트
                try:
                    if file_type == 'zip':
                        # ZIP 파일 처리
                        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                            temp_dir = f"{temp_path}_extract"
                            zip_ref.extractall(temp_dir)
                            
                            # data.pkl 찾기
                            data_pkl_path = Path(temp_dir) / "data.pkl"
                            if data_pkl_path.exists():
                                model_data = torch.load(str(data_pkl_path), map_location='cpu')
                                torch.save(model_data, model_path)
                                shutil.rmtree(temp_dir)
                                os.remove(temp_path)
                                return True
                            
                            shutil.rmtree(temp_dir)
                    else:
                        # PyTorch/Pickle 파일 처리
                        model_data = torch.load(temp_path, map_location='cpu', weights_only=True)
                        torch.save(model_data, model_path)
                        os.remove(temp_path)
                        return True
                        
                except Exception as e:
                    print(f"   ❌ {file_type} 파일 처리 실패: {e}")
                    if Path(temp_path).exists():
                        os.remove(temp_path)
                    continue
        
        return False
    
    def _try_binary_repair(self, model_path: str) -> bool:
        """바이너리 수리 시도"""
        try:
            # 파일을 바이너리로 읽기
            with open(model_path, 'rb') as f:
                data = f.read()
            
            # 파일 크기 확인
            if len(data) < 100:
                return False
            
            # 헤더 부분 제거 시도
            for i in range(0, min(1000, len(data)), 100):
                try:
                    temp_path = f"{model_path}_temp"
                    with open(temp_path, 'wb') as f:
                        f.write(data[i:])
                    
                    # 임시 파일 테스트
                    model_data = torch.load(temp_path, map_location='cpu', weights_only=True)
                    torch.save(model_data, model_path)
                    os.remove(temp_path)
                    print(f"   ✅ 바이너리 수리 성공 (오프셋: {i})")
                    return True
                    
                except Exception as e:
                    if Path(temp_path).exists():
                        os.remove(temp_path)
                    continue
            
            return False
            
        except Exception as e:
            print(f"   ❌ 바이너리 수리 실패: {e}")
            return False
    
    def _convert_to_safetensors(self, model_path: str) -> bool:
        """PyTorch 모델을 SafeTensors로 변환"""
        try:
            # 모델 로딩
            model_data = None
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
                if 'state_dict' in model_data:
                    safetensors.save_file(model_data['state_dict'], safetensors_path)
                else:
                    safetensors.save_file(model_data, safetensors_path)
            else:
                safetensors.save_file({'model': model_data}, safetensors_path)
            
            # 원본 파일 교체
            shutil.move(model_path, f"{model_path}.old")
            shutil.move(safetensors_path, model_path)
            
            return True
            
        except Exception as e:
            print(f"   ❌ SafeTensors 변환 실패: {e}")
            return False
    
    def _reconstruct_pytorch_file(self, model_path: str) -> bool:
        """PyTorch 파일 재구성"""
        try:
            # 파일을 바이너리로 읽기
            with open(model_path, 'rb') as f:
                data = f.read()
            
            # PyTorch 시그니처 찾기
            pytorch_signatures = [b'PK\x03\x04', b'pytorch', b'pickle']
            
            for signature in pytorch_signatures:
                pos = data.find(signature)
                if pos != -1:
                    # 시그니처부터 끝까지 추출
                    valid_data = data[pos:]
                    
                    # 임시 파일에 저장
                    temp_path = f"{model_path}_temp"
                    with open(temp_path, 'wb') as f:
                        f.write(valid_data)
                    
                    # 임시 파일 테스트
                    try:
                        model_data = torch.load(temp_path, map_location='cpu', weights_only=True)
                        torch.save(model_data, model_path)
                        os.remove(temp_path)
                        return True
                    except Exception as e:
                        if Path(temp_path).exists():
                            os.remove(temp_path)
                        continue
            
            return False
            
        except Exception as e:
            print(f"   ❌ PyTorch 파일 재구성 실패: {e}")
            return False
    
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
    
    def generate_report(self):
        """수정 리포트 생성"""
        report = []
        report.append("🔥 고급 AI 모델 수정 리포트")
        report.append("=" * 80)
        report.append(f"📅 수정 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append(f"📊 수정 결과:")
        report.append(f"   ✅ 성공: {len(self.fixed_models)}개")
        report.append(f"   ❌ 실패: {len(self.failed_models)}개")
        report.append("")
        
        if self.fixed_models:
            report.append("✅ 수정 완료된 모델들:")
            for model_path in self.fixed_models:
                report.append(f"   - {Path(model_path).name}")
            report.append("")
        
        if self.failed_models:
            report.append("❌ 수정 실패한 모델들:")
            for model_path in self.failed_models:
                report.append(f"   - {Path(model_path).name}")
            report.append("")
        
        return "\n".join(report)

def main():
    """메인 함수"""
    # 수정기 초기화
    fixer = AdvancedModelFixer()
    
    # 모든 문제가 있는 모델들 수정
    fixer.fix_all_problematic_models()
    
    # 수정된 모델들 검증
    fixer.verify_fixed_models()
    
    # 수정 리포트 생성
    report = fixer.generate_report()
    print(report)
    
    # 리포트 저장
    with open("advanced_ai_model_fix_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n💾 수정 리포트 저장: advanced_ai_model_fix_report.txt")
    print("\n🎉 고급 AI 모델 수정 완료!")

if __name__ == "__main__":
    from datetime import datetime
    main()
