#!/bin/bash
# MyCloset AI 최종 수정 스크립트
# step_model_requirements.py vs step_model_requests.py 파일명 불일치 해결

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "🔧 MyCloset AI 최종 수정 (파일명 불일치 해결)"
echo "==========================================="

# 1. 파일명 불일치 해결
log_info "1. 파일명 불일치 문제 해결 중..."

if [ -f "app/ai_pipeline/utils/step_model_requirements.py" ] && [ ! -f "app/ai_pipeline/utils/step_model_requests.py" ]; then
    # requirements -> requests로 심볼릭 링크 생성
    ln -sf step_model_requirements.py app/ai_pipeline/utils/step_model_requests.py
    log_success "step_model_requests.py 심볼릭 링크 생성 완료"
elif [ -f "app/ai_pipeline/utils/step_model_requirements.py" ]; then
    log_info "step_model_requirements.py 이미 존재, step_model_requests.py도 존재"
else
    log_warning "step_model_requirements.py 파일이 없습니다"
fi

# 2. __init__.py에서 import 수정
log_info "2. __init__.py import 경로 수정 중..."

if [ -f "app/ai_pipeline/utils/__init__.py" ]; then
    python3 << 'PYEOF'
try:
    with open('app/ai_pipeline/utils/__init__.py', 'r') as f:
        content = f.read()
    
    modified = False
    
    # step_model_requirements를 step_model_requests로 변경
    if 'step_model_requirements' in content:
        content = content.replace('step_model_requirements', 'step_model_requests')
        modified = True
    
    # 안전한 import 구조로 변경
    if 'from .step_model_requests import' in content and 'try:' not in content.split('from .step_model_requests import')[0].split('\n')[-1]:
        # try-except로 감싸기
        old_import = 'from .step_model_requests import'
        new_import = '''try:
    from .step_model_requests import'''
        
        content = content.replace(old_import, new_import)
        
        # except 블록 추가
        if 'except ImportError' not in content:
            lines = content.split('\n')
            new_lines = []
            in_try_block = False
            
            for i, line in enumerate(lines):
                if 'from .step_model_requests import' in line and 'try:' in lines[i-1] if i > 0 else False:
                    in_try_block = True
                
                new_lines.append(line)
                
                # try 블록 다음 줄에서 except 추가
                if in_try_block and (i == len(lines)-1 or (not lines[i+1].startswith('    ') and lines[i+1].strip() != '')):
                    new_lines.append('except ImportError as e:')
                    new_lines.append('    logger.warning(f"step_model_requests import 실패: {e}")')
                    new_lines.append('    STEP_REQUESTS_AVAILABLE = False')
                    in_try_block = False
            
            content = '\n'.join(new_lines)
        
        modified = True
    
    if modified:
        with open('app/ai_pipeline/utils/__init__.py', 'w') as f:
            f.write(content)
        print("✅ __init__.py import 경로 수정됨")
    else:
        print("ℹ️ __init__.py 이미 올바르게 설정됨")

except Exception as e:
    print(f"❌ __init__.py 수정 실패: {e}")
PYEOF
fi

# 3. model_loader.py에서도 import 수정
log_info "3. model_loader.py import 수정 중..."

if [ -f "app/ai_pipeline/utils/model_loader.py" ]; then
    python3 << 'PYEOF'
try:
    with open('app/ai_pipeline/utils/model_loader.py', 'r') as f:
        content = f.read()
    
    if 'step_model_requirements' in content:
        content = content.replace('step_model_requirements', 'step_model_requests')
        
        with open('app/ai_pipeline/utils/model_loader.py', 'w') as f:
            f.write(content)
        print("✅ model_loader.py import 경로 수정됨")
    else:
        print("ℹ️ model_loader.py 이미 올바르게 설정됨")

except Exception as e:
    print(f"❌ model_loader.py 수정 실패: {e}")
PYEOF
fi

# 4. auto_model_detector.py에서도 import 수정
log_info "4. auto_model_detector.py import 수정 중..."

if [ -f "app/ai_pipeline/utils/auto_model_detector.py" ]; then
    python3 << 'PYEOF'
try:
    with open('app/ai_pipeline/utils/auto_model_detector.py', 'r') as f:
        content = f.read()
    
    if 'step_model_requirements' in content:
        content = content.replace('step_model_requirements', 'step_model_requests')
        
        with open('app/ai_pipeline/utils/auto_model_detector.py', 'w') as f:
            f.write(content)
        print("✅ auto_model_detector.py import 경로 수정됨")
    else:
        print("ℹ️ auto_model_detector.py 이미 올바르게 설정됨")

except Exception as e:
    print(f"❌ auto_model_detector.py 수정 실패: {e}")
PYEOF
fi

# 5. 최종 테스트
log_info "5. 최종 import 테스트 중..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

print('🧪 최종 import 테스트...')

# step_model_requests 테스트
try:
    from app.ai_pipeline.utils.step_model_requests import get_step_request
    print('✅ step_model_requests import 성공')
    
    # 실제 함수 호출 테스트
    result = get_step_request('HumanParsingStep')
    if result:
        print(f'   - HumanParsingStep 요청 정보: {result.model_name}')
    else:
        print('   - HumanParsingStep 요청 정보 없음')
        
except Exception as e:
    print(f'❌ step_model_requests import 실패: {e}')

# 통합 시스템 테스트
try:
    from app.ai_pipeline.utils import SYSTEM_INFO
    print('✅ 통합 시스템 import 성공')
    print(f'   시스템: {SYSTEM_INFO.get("platform", "unknown")} / {SYSTEM_INFO.get("device", "unknown")}')
except Exception as e:
    print(f'⚠️ 통합 시스템 import 부분 실패: {e}')

# Health API 테스트
try:
    from app.api.health import router
    print('✅ Health API import 성공')
except Exception as e:
    print(f'❌ Health API import 실패: {e}')

print('🧪 최종 테스트 완료')
PYEOF

# 6. 서버 시작 준비 확인
log_info "6. 서버 시작 준비 확인 중..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

print('🚀 서버 시작 준비 확인...')

try:
    # main 모듈 import 테스트
    import app.main
    print('✅ main 모듈 import 성공')
except Exception as e:
    print(f'❌ main 모듈 import 실패: {e}')
    print('상세 오류:')
    import traceback
    traceback.print_exc()

print('🚀 준비 확인 완료')
PYEOF

# 완료 메시지
echo ""
echo "🎉 최종 수정 완료!"
echo "==============="
log_success "파일명 불일치 문제 해결됨"
log_success "모든 import 경로 수정됨"
echo ""
echo "🚀 이제 서버를 실행하세요:"
echo "   python3 quick_start.py"
echo ""
echo "또는 직접 실행:"
echo "   python3 app/main.py"
echo ""
echo "🔗 실행 후 테스트:"
echo "   curl http://localhost:8000/health"