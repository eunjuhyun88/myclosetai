#!/usr/bin/env python3
"""
MyCloset AI - conda 환경별 모델 설정
생성 시간: 2025-07-19 23:54:24
conda 모델: 11개
"""

from pathlib import Path
from typing import Dict, List

# conda 환경 정보
CONDA_ENVIRONMENTS = {
    "base": Path(r"/opt/homebrew/Caskroom/miniforge/base"),
    "mycloset-ai": Path(r"/opt/homebrew/Caskroom/miniforge/base/envs/mycloset-ai"),
    "mycloset-m3": Path(r"/opt/homebrew/Caskroom/miniforge/base/envs/mycloset-m3"),
    "mycloset-m3max": Path(r"/opt/homebrew/Caskroom/miniforge/base/envs/mycloset-m3max"),
    "tryon": Path(r"/opt/homebrew/Caskroom/miniforge/base/envs/tryon"),
    "virtual_tryon": Path(r"/opt/homebrew/Caskroom/miniforge/base/envs/virtual_tryon"),
    "vton-m3": Path(r"/opt/homebrew/Caskroom/miniforge/base/envs/vton-m3"),
}

CURRENT_CONDA_ENV = "mycloset-ai"

# conda 환경별 모델 매핑
CONDA_MODELS = {
    "base": [
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.4.1-cpu_generic_py311h4333a05_0/info/test/test/package/package_bc/test_nn_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.4.1-cpu_generic_py311h4333a05_0/info/test/test/package/package_e/test_nn_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.4.1-cpu_generic_py311h4333a05_0/info/test/test/package/package_bc/test_fx_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.1.0-cpu_generic_py311h9ea7f2d_0/info/test/test/package/package_bc/test_nn_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.1.0-cpu_generic_py311h9ea7f2d_0/info/test/test/package/package_e/test_nn_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.1.0-cpu_generic_py311h9ea7f2d_0/info/test/test/package/package_bc/test_torchscript_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.4.1-cpu_generic_py311h4333a05_0/info/test/test/package/package_bc/test_torchscript_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pytorch-2.1.0-cpu_generic_py311h9ea7f2d_0/info/test/test/package/package_bc/test_fx_module.pt",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pillow-10.1.0-py311hb9c5795_0/info/test/Tests/images/sgi_crash.bin",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pillow-11.3.0-py312h50aef2c_0/info/test/Tests/images/sgi_crash.bin",
        "/opt/homebrew/Caskroom/miniforge/base/pkgs/pillow-10.3.0-py311h0b5d0a1_0/info/test/Tests/images/sgi_crash.bin",
    ],
}

def get_conda_model_paths(env_name: str) -> List[str]:
    """conda 환경별 모델 경로 목록"""
    return CONDA_MODELS.get(env_name, [])

def get_current_env_models() -> List[str]:
    """현재 conda 환경의 모델들"""
    if CURRENT_CONDA_ENV != "None":
        return get_conda_model_paths(CURRENT_CONDA_ENV)
    return []

def list_conda_environments() -> List[str]:
    """conda 환경 목록"""
    return list(CONDA_ENVIRONMENTS.keys())

if __name__ == "__main__":
    print("🐍 MyCloset AI conda 환경 모델 설정")
    print("=" * 50)
    
    print(f"현재 환경: {CURRENT_CONDA_ENV}")
    print(f"총 환경: {len(CONDA_ENVIRONMENTS)}개")
    
    for env_name in CONDA_ENVIRONMENTS.keys():
        models = get_conda_model_paths(env_name)
        print(f"  {env_name}: {len(models)}개 모델")
