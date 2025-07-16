#!/usr/bin/env python3
import os
from pathlib import Path

def get_current_structure():
    """현재 구조 파악"""
    current = []
    for root, dirs, files in os.walk('.'):
        # .git, node_modules, venv 제외
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', 'venv', '__pycache__']]
        
        level = root.replace('.', '').count(os.sep)
        if level < 3:  # 깊이 3까지만
            indent = '  ' * level
            current.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = '  ' * (level + 1)
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    current.append(f"{subindent}{file}")
    
    return current

def get_expected_structure():
    """예상 구조"""
    return [
        "mycloset-ai/",
        "  frontend/",
        "    src/",
        "      App.tsx",
        "      main.tsx",
        "      components/",
        "      pages/",
        "      hooks/",
        "      utils/",
        "      types/",
        "    public/",
        "    package.json",
        "    tailwind.config.js",
        "  backend/",
        "    app/",
        "      main.py",
        "      api/",
        "        routes.py",
        "      core/",
        "        config.py",
        "      models/",
        "        ootd_model.py",
        "      services/",
        "        virtual_tryon.py",
        "      utils/",
        "        file_manager.py",
        "        image_utils.py",
        "    ai_models/",
        "      checkpoints/",
        "      OOTDiffusion/",
        "      VITON-HD/",
        "    static/",
        "      uploads/",
        "      results/",
        "    venv/",
        "    requirements.txt",
        "    .env",
        "  scripts/",
        "    setup.sh",
        "    download_models.py",
        "  docker/",
        "    docker-compose.yml",
        "  .gitignore",
        "  README.md",
        "  Makefile"
    ]

def compare_structures():
    """구조 비교"""
    print("🔍 MyCloset AI 구조 비교 분석")
    print("=" * 50)
    
    current = get_current_structure()
    expected = get_expected_structure()
    
    # 현재 구조에서 파일명만 추출
    current_files = set()
    for item in current:
        clean_item = item.strip().replace('/', '')
        if clean_item and not clean_item.endswith('/'):
            current_files.add(clean_item)
    
    expected_files = set()
    for item in expected:
        clean_item = item.strip().replace('/', '')
        if clean_item and not clean_item.endswith('/'):
            expected_files.add(clean_item)
    
    print("\n✅ 존재하는 중요 파일들:")
    existing = current_files & expected_files
    for file in sorted(existing):
        print(f"   ✅ {file}")
    
    print("\n❌ 누락된 중요 파일들:")
    missing = expected_files - current_files
    for file in sorted(missing):
        print(f"   ❌ {file}")
    
    print("\n📊 완성도:")
    if expected_files:
        completion = len(existing) / len(expected_files) * 100
        print(f"   {completion:.1f}% 완료 ({len(existing)}/{len(expected_files)})")

if __name__ == "__main__":
    compare_structures()
