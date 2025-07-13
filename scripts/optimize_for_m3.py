
#!/usr/bin/env python3
"""
M3 Max 특화 최적화 스크립트
"""

import torch
import psutil
import platform

def check_m3_max():
    """M3 Max 환경 확인"""
    print("🔍 시스템 환경 확인...")
    
    # 시스템 정보
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    # PyTorch MPS 지원 확인
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) 사용 가능")
        print("🚀 M3 Max 최적화 활성화됨")
        return True
    else:
        print("❌ MPS 사용 불가 - CPU 모드로 실행")
        return False

def optimize_pytorch():
    """PyTorch 최적화 설정"""
    if torch.backends.mps.is_available():
        # MPS 메모리 할당 최적화
        torch.mps.set_per_process_memory_fraction(0.8)
        
        # 메모리 효율적 어텐션 비활성화 (MPS 호환성)
        torch.backends.cuda.enable_flash_sdp(False)
        
        print("⚡ MPS 최적화 설정 완료")
    
    # 일반 최적화
    torch.set_num_threads(psutil.cpu_count())
    torch.backends.cudnn.benchmark = False  # MPS에서는 불필요
    
    print("🔧 PyTorch 최적화 완료")

def check_memory_requirements():
    """메모리 요구사항 확인"""
    available_memory = psutil.virtual_memory().available / (1024**3)
    required_memory = 8.0  # 최소 8GB 권장
    
    if available_memory >= required_memory:
        print(f"✅ 메모리 충족: {available_memory:.1f}GB 사용 가능")
        return True
    else:
        print(f"⚠️  메모리 부족: {available_memory:.1f}GB 사용 가능 (최소 {required_memory}GB 권장)")
        return False

if __name__ == "__main__":
    print("🍎 M3 Max 최적화 스크립트")
    print("=" * 40)
    
    m3_available = check_m3_max()
    memory_ok = check_memory_requirements()
    
    if m3_available and memory_ok:
        optimize_pytorch()
        print("\n🎉 M3 Max 최적화 완료!")
        print("💡 권장사항:")
        print("  - 고품질 모드: 메모리 16GB+ 환경에서 사용")
        print("  - 균형 모드: 일반적인 사용에 권장")
        print("  - 빠른 모드: 메모리 제한이 있는 환경")
    else:
        print("\n⚠️  최적화 제한사항이 있습니다.")
        print("CPU 모드로 실행하거나 메모리를 확보해주세요.")
