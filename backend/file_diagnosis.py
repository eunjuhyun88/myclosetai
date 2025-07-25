#!/usr/bin/env python3
"""
🔍 MyCloset AI - 모델 파일 진단 및 복구 도구
===============================================================================

문제 진단:
RuntimeError: Expected hasRecord("version") to be true, but got false.

가능한 원인들:
1. 파일이 손상됨 (다운로드 중 문제)
2. 잘못된 포맷 (PyTorch가 아닌 다른 프레임워크)
3. PyTorch 버전 호환성 문제
4. 파일이 압축됨 (.gz, .zip 등)
5. 텍스트 파일이거나 다른 형식

해결 방법들을 단계별로 시도
"""

import os
import sys
from pathlib import Path
import subprocess
import magic  # python-magic 라이브러리
import gzip
import zipfile
import tarfile

class ModelFileDiagnostic:
    """모델 파일 진단 도구"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.diagnosis = {}
        
    def run_full_diagnosis(self):
        """전체 진단 실행"""
        print(f"🔍 파일 진단 시작: {self.file_path}")
        print("="*60)
        
        # 1. 기본 파일 정보
        self._check_basic_info()
        
        # 2. 파일 타입 확인
        self._check_file_type()
        
        # 3. 매직 넘버 확인
        self._check_magic_bytes()
        
        # 4. 압축 파일 여부 확인
        self._check_compression()
        
        # 5. PyTorch 로딩 시도 (다양한 방법)
        self._try_pytorch_loading()
        
        # 6. 복구 제안
        self._suggest_solutions()
        
        return self.diagnosis
    
    def _check_basic_info(self):
        """기본 파일 정보 확인"""
        try:
            stat = self.file_path.stat()
            self.diagnosis['basic_info'] = {
                'exists': self.file_path.exists(),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024**2), 2),
                'readable': os.access(self.file_path, os.R_OK),
                'file_extension': self.file_path.suffix
            }
            
            print(f"📁 파일 정보:")
            print(f"  크기: {self.diagnosis['basic_info']['size_mb']} MB")
            print(f"  확장자: {self.diagnosis['basic_info']['file_extension']}")
            print(f"  읽기 권한: {self.diagnosis['basic_info']['readable']}")
            
        except Exception as e:
            print(f"❌ 기본 정보 확인 실패: {e}")
            self.diagnosis['basic_info'] = {'error': str(e)}
    
    def _check_file_type(self):
        """file 명령어로 파일 타입 확인"""
        try:
            result = subprocess.run(['file', str(self.file_path)], 
                                 capture_output=True, text=True)
            file_type = result.stdout.strip()
            
            self.diagnosis['file_type'] = file_type
            print(f"🔍 File 명령어 결과:")
            print(f"  {file_type}")
            
        except Exception as e:
            print(f"⚠️ file 명령어 실패: {e}")
            self.diagnosis['file_type'] = f"Error: {e}"
    
    def _check_magic_bytes(self):
        """파일의 매직 바이트 확인"""
        try:
            with open(self.file_path, 'rb') as f:
                magic_bytes = f.read(16)
            
            self.diagnosis['magic_bytes'] = {
                'hex': magic_bytes.hex(),
                'first_4_bytes': magic_bytes[:4],
                'printable': ''.join(chr(b) if 32 <= b <= 126 else '.' for b in magic_bytes)
            }
            
            print(f"🔮 매직 바이트:")
            print(f"  Hex: {magic_bytes.hex()}")
            print(f"  ASCII: {self.diagnosis['magic_bytes']['printable']}")
            
            # 알려진 형식 확인
            self._identify_format_by_magic(magic_bytes)
            
        except Exception as e:
            print(f"❌ 매직 바이트 확인 실패: {e}")
            self.diagnosis['magic_bytes'] = {'error': str(e)}
    
    def _identify_format_by_magic(self, magic_bytes):
        """매직 바이트로 형식 식별"""
        magic_signatures = {
            b'PK\x03\x04': 'ZIP archive',
            b'PK\x05\x06': 'ZIP archive (empty)',
            b'\x1f\x8b': 'GZIP compressed',
            b'\x42\x5a': 'BZIP2 compressed',
            b'\x50\x4b': 'ZIP/DOCX/XLSX/etc',
            b'\x89PNG': 'PNG image',
            b'\xff\xd8\xff': 'JPEG image',
            b'{\x0a\x20\x20': 'JSON text file',
            b'#!/usr/bin': 'Shell script',
        }
        
        format_found = None
        for signature, format_name in magic_signatures.items():
            if magic_bytes.startswith(signature):
                format_found = format_name
                break
        
        if format_found:
            print(f"  🎯 감지된 형식: {format_found}")
            self.diagnosis['detected_format'] = format_found
        else:
            print(f"  ❓ 알 수 없는 형식")
            self.diagnosis['detected_format'] = 'unknown'
    
    def _check_compression(self):
        """압축 파일 여부 확인"""
        compression_checks = []
        
        # GZIP 확인
        try:
            with gzip.open(self.file_path, 'rb') as f:
                f.read(10)  # 작은 데이터 읽기 시도
            compression_checks.append('gzip')
            print("  ✅ GZIP 형식으로 읽기 가능")
        except:
            pass
        
        # ZIP 확인
        try:
            with zipfile.ZipFile(self.file_path, 'r') as z:
                files = z.namelist()
            compression_checks.append('zip')
            print(f"  ✅ ZIP 아카이브 (파일 {len(files)}개)")
        except:
            pass
        
        # TAR 확인
        try:
            with tarfile.open(self.file_path, 'r') as t:
                members = t.getmembers()
            compression_checks.append('tar')
            print(f"  ✅ TAR 아카이브 (멤버 {len(members)}개)")
        except:
            pass
        
        self.diagnosis['compression'] = compression_checks
        
        if not compression_checks:
            print("  📄 압축되지 않은 파일")
    
    def _try_pytorch_loading(self):
        """다양한 방법으로 PyTorch 로딩 시도"""
        print(f"\n🔧 PyTorch 로딩 시도:")
        loading_attempts = {}
        
        # 1. 기본 로딩
        try:
            import torch
            data = torch.load(self.file_path, map_location='cpu')
            loading_attempts['basic'] = {'success': True, 'type': type(data).__name__}
            print("  ✅ 기본 torch.load() 성공")
        except Exception as e:
            loading_attempts['basic'] = {'success': False, 'error': str(e)}
            print(f"  ❌ 기본 torch.load() 실패: {e}")
        
        # 2. weights_only=True로 시도 (PyTorch 1.13+)
        try:
            import torch
            data = torch.load(self.file_path, map_location='cpu', weights_only=True)
            loading_attempts['weights_only'] = {'success': True, 'type': type(data).__name__}
            print("  ✅ weights_only=True 로딩 성공")
        except Exception as e:
            loading_attempts['weights_only'] = {'success': False, 'error': str(e)}
            print(f"  ❌ weights_only=True 실패: {e}")
        
        # 3. pickle_module 지정
        try:
            import torch
            import pickle
            data = torch.load(self.file_path, map_location='cpu', pickle_module=pickle)
            loading_attempts['pickle_module'] = {'success': True, 'type': type(data).__name__}
            print("  ✅ pickle_module 지정 성공")
        except Exception as e:
            loading_attempts['pickle_module'] = {'success': False, 'error': str(e)}
            print(f"  ❌ pickle_module 지정 실패: {e}")
        
        # 4. 압축 해제 후 시도
        if 'gzip' in self.diagnosis.get('compression', []):
            try:
                import torch
                import gzip
                with gzip.open(self.file_path, 'rb') as f:
                    data = torch.load(f, map_location='cpu')
                loading_attempts['gzip_decompressed'] = {'success': True, 'type': type(data).__name__}
                print("  ✅ GZIP 압축 해제 후 로딩 성공")
            except Exception as e:
                loading_attempts['gzip_decompressed'] = {'success': False, 'error': str(e)}
                print(f"  ❌ GZIP 압축 해제 후 실패: {e}")
        
        self.diagnosis['pytorch_loading'] = loading_attempts
    
    def _suggest_solutions(self):
        """해결책 제안"""
        print(f"\n💡 해결책 제안:")
        solutions = []
        
        # 파일 손상 확인
        if self.diagnosis['basic_info']['size_mb'] < 1:
            solutions.append("파일이 너무 작습니다. 다운로드가 완료되지 않았을 수 있습니다.")
        
        # 압축 파일인 경우
        if self.diagnosis.get('compression'):
            solutions.append("압축 파일로 감지됨. 압축을 해제한 후 다시 시도하세요.")
        
        # 알려진 형식이 아닌 경우
        if self.diagnosis.get('detected_format') == 'unknown':
            solutions.append("PyTorch 체크포인트가 아닐 수 있습니다. 파일 출처를 확인하세요.")
        
        # PyTorch 로딩이 모두 실패한 경우
        pytorch_attempts = self.diagnosis.get('pytorch_loading', {})
        if not any(attempt.get('success', False) for attempt in pytorch_attempts.values()):
            solutions.extend([
                "파일이 손상되었을 가능성이 높습니다.",
                "원본 파일을 다시 다운로드하세요.",
                "다른 PyTorch 버전으로 시도해보세요.",
                "파일 제공자에게 올바른 형식인지 확인하세요."
            ])
        
        for i, solution in enumerate(solutions, 1):
            print(f"  {i}. {solution}")
        
        self.diagnosis['suggested_solutions'] = solutions

# ==============================================
# 🔧 복구 도구들
# ==============================================

def try_decompress_file(file_path: Path) -> bool:
    """압축된 파일 압축 해제 시도"""
    print(f"🔄 압축 해제 시도: {file_path}")
    
    # GZIP 시도
    try:
        with gzip.open(file_path, 'rb') as f_in:
            decompressed_path = file_path.with_suffix(file_path.suffix + '.decompressed')
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"✅ GZIP 압축 해제 완료: {decompressed_path}")
        return True
    except:
        pass
    
    # ZIP 시도
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            extract_dir = file_path.parent / (file_path.stem + '_extracted')
            z.extractall(extract_dir)
        print(f"✅ ZIP 압축 해제 완료: {extract_dir}")
        return True
    except:
        pass
    
    print("❌ 압축 해제 실패")
    return False

def check_file_integrity(file_path: Path) -> bool:
    """파일 무결성 확인"""
    try:
        # 파일을 끝까지 읽을 수 있는지 확인
        with open(file_path, 'rb') as f:
            chunk_size = 1024 * 1024  # 1MB 청크
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
        print(f"✅ 파일 무결성 확인 완료: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 파일 무결성 확인 실패: {e}")
        return False

# ==============================================
# 🔧 실행 함수
# ==============================================

def diagnose_model_files():
    """모델 파일들 진단"""
    model_files = [
        "ai_models/step_01_human_parsing/graphonomy.pth",
        "ai_models/step_01_human_parsing/exp-schp-201908301523-atr.pth", 
        "ai_models/step_01_human_parsing/atr_model.pth",
        "ai_models/step_01_human_parsing/lip_model.pth"
    ]
    
    print("🔍 MyCloset AI - 모델 파일 진단 도구")
    print("="*60)
    
    for model_file in model_files:
        file_path = Path(model_file)
        
        if not file_path.exists():
            print(f"\n❌ 파일 없음: {model_file}")
            continue
        
        print(f"\n" + "="*60)
        diagnostic = ModelFileDiagnostic(model_file)
        results = diagnostic.run_full_diagnosis()
        
        # 파일 무결성 확인
        print(f"\n🔒 무결성 확인:")
        check_file_integrity(file_path)
        
        # 압축 해제 시도 (필요한 경우)
        if results.get('compression'):
            print(f"\n🔄 압축 해제 시도:")
            try_decompress_file(file_path)

if __name__ == "__main__":
    try:
        diagnose_model_files()
    except ImportError as e:
        print(f"❌ 필요한 라이브러리 없음: {e}")
        print("다음 명령어로 설치하세요:")
        print("pip install python-magic-bin")  # Windows용
        print("pip install python-magic")      # Unix용