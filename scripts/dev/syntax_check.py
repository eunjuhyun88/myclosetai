#!/usr/bin/env python3
"""빠른 Python 구문 검사기"""

import sys
from pathlib import Path

def check_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, str(file_path), 'exec')
        print(f"✅ {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}:{e.lineno} - {e.msg}")
        return False
    except Exception as e:
        print(f"⚠️ {file_path} - {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        check_syntax(file_path)
    else:
        print("사용법: python syntax_check.py <파일경로>")
