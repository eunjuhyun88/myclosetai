#!/usr/bin/env python3
"""
메모리 사용량 분석 스크립트
51GB 메모리 사용량의 원인을 찾습니다.
"""

import psutil
import os
import sys

def analyze_memory_usage():
    """메모리 사용량 분석"""
    print("🔥 메모리 사용량 분석 시작")
    
    # 전체 시스템 메모리
    memory = psutil.virtual_memory()
    print(f"\n📊 전체 시스템 메모리:")
    print(f"   - 전체: {memory.total / (1024**3):.1f}GB")
    print(f"   - 사용 중: {memory.used / (1024**3):.1f}GB")
    print(f"   - 사용률: {memory.percent}%")
    print(f"   - 가용: {memory.available / (1024**3):.1f}GB")
    
    # 현재 Python 프로세스 메모리
    current_process = psutil.Process(os.getpid())
    print(f"\n📊 현재 Python 프로세스:")
    print(f"   - PID: {current_process.pid}")
    print(f"   - 메모리: {current_process.memory_info().rss / (1024**3):.2f}GB")
    print(f"   - 가상 메모리: {current_process.memory_info().vms / (1024**3):.2f}GB")
    
    # 상위 메모리 사용 프로세스
    print(f"\n📊 상위 메모리 사용 프로세스:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            memory_info = proc.info['memory_info']
            if memory_info is not None:
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': memory_info.rss / (1024**2),
                    'cmdline': ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else ''
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # 메모리 사용량 순으로 정렬
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    for i, proc in enumerate(processes[:15]):
        print(f"   {i+1:2d}. {proc['name']:<20} {proc['memory_mb']:8.1f}MB (PID: {proc['pid']})")
        if proc['cmdline']:
            print(f"       └─ {proc['cmdline']}")
    
    # Python 프로세스들
    print(f"\n📊 Python 프로세스들:")
    python_processes = [p for p in processes if 'python' in p['name'].lower()]
    for proc in python_processes:
        print(f"   - {proc['name']}: {proc['memory_mb']:.1f}MB (PID: {proc['pid']})")
        if proc['cmdline']:
            print(f"     └─ {proc['cmdline']}")
    
    # 메모리 사용량이 큰 프로세스들 (1GB 이상)
    large_processes = [p for p in processes if p['memory_mb'] > 1024]
    print(f"\n📊 대용량 메모리 프로세스 (1GB 이상):")
    total_large_memory = 0
    for proc in large_processes:
        memory_gb = proc['memory_mb'] / 1024
        total_large_memory += memory_gb
        print(f"   - {proc['name']}: {memory_gb:.1f}GB (PID: {proc['pid']})")
    
    print(f"\n📊 대용량 프로세스 총 메모리: {total_large_memory:.1f}GB")
    
    # 메모리 사용량 분석
    print(f"\n📊 메모리 사용량 분석:")
    print(f"   - 전체 사용 중: {memory.used / (1024**3):.1f}GB")
    print(f"   - 대용량 프로세스: {total_large_memory:.1f}GB")
    print(f"   - 기타 프로세스: {(memory.used / (1024**3)) - total_large_memory:.1f}GB")
    
    # AI 모델 관련 메모리 확인
    print(f"\n📊 AI 모델 관련 메모리 확인:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   - CUDA 메모리 할당: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")
            print(f"   - CUDA 메모리 캐시: {torch.cuda.memory_reserved() / (1024**3):.2f}GB")
        else:
            print(f"   - CUDA 사용 불가")
    except ImportError:
        print(f"   - PyTorch 미설치")
    
    try:
        import torch.mps
        if torch.backends.mps.is_available():
            print(f"   - MPS 사용 가능")
            # MPS 메모리 정보는 직접적으로 확인하기 어려움
        else:
            print(f"   - MPS 사용 불가")
    except:
        print(f"   - MPS 확인 불가")

if __name__ == "__main__":
    analyze_memory_usage() 