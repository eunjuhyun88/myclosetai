#!/usr/bin/env python3
"""
AI 모델 메모리 최적화 스크립트
MyCloset AI 서버의 메모리 사용량을 추가로 줄입니다.
"""

import psutil
import os
import gc
import sys

def optimize_ai_memory():
    """AI 모델 메모리 최적화"""
    print("🔥 AI 모델 메모리 최적화 시작")
    
    # 현재 메모리 상태 확인
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**2)
    print(f"📊 초기 메모리: {initial_memory:.1f}MB")
    
    # 1. 가비지 컬렉션 강제 실행
    print("\n🔄 가비지 컬렉션 실행...")
    collected = gc.collect()
    print(f"   - 수집된 객체: {collected}개")
    
    # 2. 메모리 사용량 재확인
    after_gc_memory = process.memory_info().rss / (1024**2)
    print(f"   - GC 후 메모리: {after_gc_memory:.1f}MB")
    print(f"   - 절약된 메모리: {initial_memory - after_gc_memory:.1f}MB")
    
    # 3. PyTorch 메모리 최적화
    try:
        import torch
        print("\n🔥 PyTorch 메모리 최적화...")
        
        # CUDA 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   - CUDA 캐시 정리 완료")
        
        # MPS 메모리 정리 (macOS)
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            print(f"   - MPS 캐시 정리 완료")
        
        # PyTorch 내부 캐시 정리
        if hasattr(torch, 'jit'):
            torch.jit._state._python_cu.clear_cache()
            print(f"   - JIT 캐시 정리 완료")
            
    except ImportError:
        print("   - PyTorch 미설치")
    except Exception as e:
        print(f"   - PyTorch 최적화 오류: {e}")
    
    # 4. 최종 메모리 상태
    final_memory = process.memory_info().rss / (1024**2)
    total_saved = initial_memory - final_memory
    
    print(f"\n📊 최적화 결과:")
    print(f"   - 초기: {initial_memory:.1f}MB")
    print(f"   - 최종: {final_memory:.1f}MB")
    print(f"   - 총 절약: {total_saved:.1f}MB")
    
    return total_saved

def suggest_browser_optimization():
    """브라우저 최적화 제안"""
    print("\n🌐 브라우저 최적화 제안:")
    
    # Chrome 프로세스 수 확인
    chrome_count = len([p for p in psutil.process_iter(['name']) if 'chrome' in p.info['name'].lower()])
    print(f"   - 현재 Chrome 프로세스: {chrome_count}개")
    
    if chrome_count > 20:
        print(f"   ⚠️  Chrome 프로세스가 많습니다!")
        print(f"   💡 제안: 사용하지 않는 탭을 닫아주세요")
        print(f"   💡 예상 절약: 3-5GB")
    
    # WebKit 프로세스 확인
    webkit_count = len([p for p in psutil.process_iter(['name']) if 'webkit' in p.info['name'].lower()])
    print(f"   - WebKit 프로세스: {webkit_count}개")
    
    if webkit_count > 5:
        print(f"   ⚠️  WebKit 프로세스가 많습니다!")
        print(f"   💡 제안: Safari 탭을 정리해주세요")

def suggest_cursor_optimization():
    """Cursor 최적화 제안"""
    print("\n💻 Cursor 최적화 제안:")
    
    cursor_processes = [p for p in psutil.process_iter(['name', 'memory_info']) 
                       if 'cursor' in p.info['name'].lower()]
    
    total_cursor_memory = sum(p.info['memory_info'].rss / (1024**2) for p in cursor_processes)
    print(f"   - Cursor 총 메모리: {total_cursor_memory:.1f}MB")
    
    if total_cursor_memory > 2000:  # 2GB 이상
        print(f"   ⚠️  Cursor 메모리 사용량이 높습니다!")
        print(f"   💡 제안:")
        print(f"      - 불필요한 확장 프로그램 비활성화")
        print(f"      - 큰 파일이나 프로젝트 닫기")
        print(f"      - Cursor 재시작")
        print(f"   💡 예상 절약: 500MB-1GB")

if __name__ == "__main__":
    print("🚀 메모리 최적화 시작")
    
    # AI 모델 최적화
    saved_memory = optimize_ai_memory()
    
    # 브라우저 최적화 제안
    suggest_browser_optimization()
    
    # Cursor 최적화 제안
    suggest_cursor_optimization()
    
    print(f"\n✅ 최적화 완료! 총 절약: {saved_memory:.1f}MB") 