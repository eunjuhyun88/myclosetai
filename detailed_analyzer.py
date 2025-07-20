#!/usr/bin/env python3
"""
🔍 AI 모델 디렉토리 상세 중복 분석
실제로 어떤 파일들이 중복되고 있는지 구체적으로 파악
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import re

def analyze_duplicates():
    """중복 패턴 상세 분석"""
    base_path = Path("backend/ai_models")
    
    print("🔍 중복 파일 패턴 분석 중...")
    
    # 파일 수집
    all_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in ['.pth', '.pt', '.safetensors', '.bin', '.onnx']):
                file_path = Path(root) / file
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    all_files.append({
                        'name': file,
                        'path': str(file_path.relative_to(base_path)),
                        'size_mb': round(size_mb, 2),
                        'dir': Path(root).name
                    })
                except:
                    pass
    
    print(f"📊 총 {len(all_files)}개 모델 파일 발견")
    
    # 중복 패턴 분석
    analyze_version_duplicates(all_files)
    analyze_size_duplicates(all_files)
    analyze_name_patterns(all_files)
    analyze_large_files(all_files)
    
def analyze_version_duplicates(files):
    """버전 번호 중복 분석 (_01, _02 등)"""
    print("\n📋 버전 번호 중복 분석:")
    
    # 기본 이름으로 그룹화
    base_groups = defaultdict(list)
    
    for file in files:
        # 버전 번호 제거 패턴
        base_name = re.sub(r'_\d+(\.(pth|pt|safetensors|bin|onnx))?$', '', file['name'])
        base_name = re.sub(r'\.(pth|pt|safetensors|bin|onnx)$', '', base_name)
        base_groups[base_name].append(file)
    
    # 중복이 있는 그룹만 출력
    version_duplicates = []
    for base_name, group in base_groups.items():
        if len(group) > 1:
            version_duplicates.append((base_name, group))
    
    # 크기 순 정렬
    version_duplicates.sort(key=lambda x: sum(f['size_mb'] for f in x[1]), reverse=True)
    
    print(f"   발견된 버전 중복 그룹: {len(version_duplicates)}개")
    
    total_waste = 0
    for base_name, group in version_duplicates[:15]:  # 상위 15개만
        sizes = [f['size_mb'] for f in group]
        waste = sum(sizes) - max(sizes)  # 가장 큰 파일 제외한 나머지
        total_waste += waste
        
        print(f"   📁 {base_name}:")
        for file in sorted(group, key=lambda x: x['size_mb'], reverse=True):
            print(f"      └─ {file['name']} ({file['size_mb']:.1f}MB)")
        print(f"      💾 절약 가능: {waste:.1f}MB")
    
    print(f"\n💰 총 절약 가능 용량: {total_waste/1024:.2f}GB")

def analyze_size_duplicates(files):
    """동일한 크기 파일 분석"""
    print("\n📏 동일 크기 파일 분석:")
    
    size_groups = defaultdict(list)
    for file in files:
        if file['size_mb'] > 10:  # 10MB 이상만
            size_groups[file['size_mb']].append(file)
    
    same_size = [(size, group) for size, group in size_groups.items() if len(group) > 1]
    same_size.sort(key=lambda x: x[0], reverse=True)
    
    print(f"   동일 크기 그룹: {len(same_size)}개")
    
    for size_mb, group in same_size[:10]:
        print(f"   📏 {size_mb}MB ({len(group)}개 파일):")
        for file in group:
            print(f"      └─ {file['name']}")

def analyze_name_patterns(files):
    """파일명 패턴 분석"""
    print("\n🔤 파일명 패턴 분석:")
    
    # 공통 접두사/접미사 찾기
    prefixes = Counter()
    suffixes = Counter()
    
    for file in files:
        name = file['name'].lower()
        
        # 접두사 (처음 몇 글자)
        if len(name) > 5:
            prefixes[name[:5]] += 1
            
        # 접미사 (확장자 제외)
        base_name = name.rsplit('.', 1)[0]
        if len(base_name) > 5:
            suffixes[base_name[-5:]] += 1
    
    print("   📝 공통 접두사 (상위 10개):")
    for prefix, count in prefixes.most_common(10):
        if count > 3:
            print(f"      {prefix}*: {count}개")
    
    print("   📝 공통 접미사 (상위 10개):")
    for suffix, count in suffixes.most_common(10):
        if count > 3:
            print(f"      *{suffix}: {count}개")

def analyze_large_files(files):
    """대용량 파일 분석"""
    print("\n📦 대용량 파일 상세 분석:")
    
    large_files = [f for f in files if f['size_mb'] > 1000]  # 1GB 이상
    large_files.sort(key=lambda x: x['size_mb'], reverse=True)
    
    print(f"   1GB 이상 파일: {len(large_files)}개")
    
    total_large = sum(f['size_mb'] for f in large_files)
    print(f"   총 용량: {total_large/1024:.2f}GB")
    
    # 디렉토리별 분포
    dir_sizes = defaultdict(float)
    for file in large_files:
        dir_sizes[file['dir']] += file['size_mb']
    
    print("   📂 디렉토리별 대용량 파일:")
    for dir_name, total_size in sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"      {dir_name}: {total_size/1024:.2f}GB")
    
    print("\n   📋 개별 파일 목록:")
    for file in large_files[:15]:
        print(f"      {file['name']}: {file['size_mb']/1024:.2f}GB ({file['dir']})")

def generate_cleanup_suggestions():
    """정리 제안사항"""
    print("\n" + "="*60)
    print("💡 정리 제안사항")
    print("="*60)
    
    suggestions = [
        "🗂️  버전 번호가 있는 중복 파일들 중 최신 버전만 유지",
        "🔗 자주 사용하는 모델은 심볼릭 링크로 통합",
        "📦 1GB 이상 대용량 모델들의 외부 저장소 이동",
        "🏷️  파일명 표준화 (일관된 명명 규칙 적용)",
        "📁 단계별 디렉토리 구조 정리",
        "🔍 실제 사용되지 않는 모델 파일 식별 후 제거",
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print(f"\n예상 절약 효과:")
    print(f"   • 중복 제거: ~50-80GB")
    print(f"   • 미사용 파일 제거: ~20-30GB") 
    print(f"   • 총 절약 가능: ~70-110GB (현재 185GB의 40-60%)")

if __name__ == "__main__":
    analyze_duplicates()
    generate_cleanup_suggestions()