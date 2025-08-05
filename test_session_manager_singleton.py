#!/usr/bin/env python3
"""
SessionManager 싱글톤 패턴 테스트 스크립트
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_session_manager_singleton():
    """SessionManager 싱글톤 패턴 테스트"""
    try:
        print("🧪 SessionManager 싱글톤 패턴 테스트 시작")
        
        # SessionManager 모듈 import
        from backend.app.core.session_manager import (
            get_session_manager, 
            SessionManager,
            test_session_manager_singleton
        )
        
        print("✅ SessionManager 모듈 import 성공")
        
        # 내장 테스트 함수 실행
        result = test_session_manager_singleton()
        
        if result:
            print("🎉 SessionManager 싱글톤 패턴 테스트 성공!")
            return True
        else:
            print("❌ SessionManager 싱글톤 패턴 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ SessionManager 싱글톤 패턴 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_imports():
    """여러 곳에서 import할 때 싱글톤이 유지되는지 테스트"""
    try:
        print("\n🧪 다중 import 싱글톤 테스트 시작")
        
        # 첫 번째 import
        from backend.app.core.session_manager import get_session_manager as get_sm1
        instance1 = get_sm1()
        print(f"✅ 첫 번째 import 인스턴스: {id(instance1)}")
        
        # 두 번째 import (다른 이름으로)
        from backend.app.core.session_manager import get_session_manager as get_sm2
        instance2 = get_sm2()
        print(f"✅ 두 번째 import 인스턴스: {id(instance2)}")
        
        # 직접 SessionManager 클래스 사용
        from backend.app.core.session_manager import SessionManager
        instance3 = SessionManager()
        print(f"✅ 직접 클래스 인스턴스: {id(instance3)}")
        
        # 인스턴스 ID 비교
        if id(instance1) == id(instance2) == id(instance3):
            print("🎉 다중 import 싱글톤 테스트 성공!")
            print(f"   - 모든 인스턴스가 동일: {id(instance1)}")
            return True
        else:
            print("❌ 다중 import 싱글톤 테스트 실패!")
            print(f"   - instance1: {id(instance1)}")
            print(f"   - instance2: {id(instance2)}")
            print(f"   - instance3: {id(instance3)}")
            return False
            
    except Exception as e:
        print(f"❌ 다중 import 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_data_persistence():
    """세션 데이터가 싱글톤 인스턴스에서 유지되는지 테스트"""
    try:
        print("\n🧪 세션 데이터 지속성 테스트 시작")
        
        from backend.app.core.session_manager import get_session_manager
        
        # 첫 번째 인스턴스에서 세션 추가
        instance1 = get_session_manager()
        instance1.sessions['test_session_1'] = {'data': 'test1', 'id': 'test1'}
        print(f"✅ 첫 번째 인스턴스에 세션 추가: {len(instance1.sessions)}개")
        
        # 두 번째 인스턴스에서 세션 확인
        instance2 = get_session_manager()
        print(f"✅ 두 번째 인스턴스 세션 수: {len(instance2.sessions)}개")
        
        # 세션 데이터 확인
        if 'test_session_1' in instance2.sessions:
            print("🎉 세션 데이터 지속성 테스트 성공!")
            print(f"   - 세션 데이터: {instance2.sessions['test_session_1']}")
            return True
        else:
            print("❌ 세션 데이터 지속성 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 세션 데이터 지속성 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 SessionManager 싱글톤 패턴 종합 테스트 시작")
    print("=" * 60)
    
    # 테스트 실행
    test1_result = test_session_manager_singleton()
    test2_result = test_multiple_imports()
    test3_result = test_session_data_persistence()
    
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약:")
    print(f"   - 기본 싱글톤 테스트: {'✅ 성공' if test1_result else '❌ 실패'}")
    print(f"   - 다중 import 테스트: {'✅ 성공' if test2_result else '❌ 실패'}")
    print(f"   - 데이터 지속성 테스트: {'✅ 성공' if test3_result else '❌ 실패'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 모든 테스트 통과! SessionManager 싱글톤 패턴이 정상 작동합니다.")
        sys.exit(0)
    else:
        print("\n❌ 일부 테스트 실패! SessionManager 싱글톤 패턴에 문제가 있습니다.")
        sys.exit(1) 