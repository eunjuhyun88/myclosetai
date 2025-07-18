# backend/apply_warmup_patch.py
"""
🔧 main.py에 워밍업 패치 자동 적용 스크립트
"""

import os
import re
from pathlib import Path

def apply_warmup_patch_to_main():
    """main.py에 워밍업 패치 적용"""
    
    main_py_path = Path(__file__).parent / 'app' / 'main.py'
    
    if not main_py_path.exists():
        print(f"❌ main.py 파일을 찾을 수 없습니다: {main_py_path}")
        return False
    
    try:
        # 현재 main.py 읽기
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 워밍업 비활성화 import 추가
        disable_import = "from app.core.disable_warmup import disable_warmup_globally"
        safe_import = "from app.utils.safe_warmup import safe_warmup, get_warmup_status"
        
        # import 섹션 찾기
        import_section = ""
        lines = content.split('\n')
        insert_index = 0
        
        # 마지막 import 라인 찾기
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                insert_index = i + 1
        
        # 패치가 이미 적용되었는지 확인
        if disable_import not in content:
            lines.insert(insert_index, disable_import)
            print("✅ 워밍업 비활성화 import 추가")
        
        if safe_import not in content:
            lines.insert(insert_index + 1, safe_import)
            print("✅ 안전한 워밍업 import 추가")
        
        # 2. FastAPI 앱 생성 이전에 패치 적용
        content = '\n'.join(lines)
        
        app_creation_pattern = r'(app = FastAPI\(.*?\))'
        
        if re.search(app_creation_pattern, content, re.DOTALL):
            # 워밍업 비활성화 코드 삽입
            patch_code = """
# 🔧 워밍업 오류 방지 패치 적용
try:
    disable_warmup_globally()
    logger.info("✅ 워밍업 비활성화 패치 적용 완료")
except Exception as e:
    logger.warning(f"⚠️ 워밍업 패치 적용 실패: {e}")

"""
            
            if "disable_warmup_globally()" not in content:
                content = re.sub(
                    app_creation_pattern,
                    patch_code + r'\1',
                    content,
                    flags=re.DOTALL
                )
                print("✅ 워밍업 비활성화 코드 삽입")
        
        # 3. 새로운 안전한 워밍업 엔드포인트 추가
        warmup_endpoint = '''
@app.get("/api/warmup/status")
async def get_warmup_status_endpoint():
    """워밍업 상태 조회 엔드포인트"""
    try:
        status = get_warmup_status()
        return {
            "success": True,
            "warmup_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/warmup/safe-test")
async def safe_warmup_test():
    """안전한 워밍업 테스트 엔드포인트"""
    try:
        # 테스트 객체들
        test_objects = {
            "dict_test": {"warmup": lambda: "dict warmup success"},
            "none_test": None,
            "callable_test": lambda: "callable success"
        }
        
        results = {}
        for name, obj in test_objects.items():
            success = safe_warmup(obj, name)
            results[name] = success
        
        return {
            "success": True,
            "test_results": results,
            "warmup_status": get_warmup_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
'''
        
        # 엔드포인트가 없으면 추가
        if "get_warmup_status_endpoint" not in content:
            # 파일 끝에 추가
            content += warmup_endpoint
            print("✅ 안전한 워밍업 엔드포인트 추가")
        
        # 4. 변경사항이 있으면 파일 저장
        if content != original_content:
            with open(main_py_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ main.py 패치 적용 완료: {main_py_path}")
            return True
        else:
            print("ℹ️ main.py에 이미 패치가 적용되어 있습니다")
            return True
    
    except Exception as e:
        print(f"❌ main.py 패치 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔧 main.py 워밍업 패치 적용 시작...")
    success = apply_warmup_patch_to_main()
    
    if success:
        print("\n🎉 패치 적용 완료!")
        print("\n📋 다음 단계:")
        print("1. python app/main.py  # 서버 재시작")
        print("2. curl http://localhost:8000/api/warmup/status  # 상태 확인")
        print("3. curl -X POST http://localhost:8000/api/warmup/safe-test  # 테스트")
    else:
        print("\n❌ 패치 적용 실패. 수동으로 코드를 확인해주세요.")