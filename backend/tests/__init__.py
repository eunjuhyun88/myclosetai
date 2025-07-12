"""
테스트 모듈
단위 테스트 및 통합 테스트
"""

import pytest
import asyncio
from typing import Generator

# 테스트 설정
TEST_CONFIG = {
    "test_mode": True,
    "use_mock_models": True,
    "temp_dir": "tests/temp",
    "fixtures_dir": "tests/fixtures"
}
