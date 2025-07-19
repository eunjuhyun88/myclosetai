#!/bin/bash
# 🔍 MyCloset AI 실시간 로그 모니터링 스크립트
# 이미지 처리 과정 실시간 추적

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 로그 필터링 함수들
filter_errors() {
    grep -E "(ERROR|ERRO|Failed|failed|Exception|Traceback|❌)" --color=always
}

filter_warnings() {
    grep -E "(WARNING|WARN|⚠️)" --color=always
}

filter_success() {
    grep -E "(SUCCESS|✅|Step.*완료|완료)" --color=always
}

filter_image_processing() {
    grep -E "(이미지|image|Step [1-8]|fitted_image|base64|PIL|numpy)" --color=always
}

filter_session() {
    grep -E "(Session|session|세션)" --color=always
}

filter_api() {
    grep -E "(API|POST|GET|/api/)" --color=always
}

show_help() {
    echo "🔍 MyCloset AI 로그 모니터링 도구"
    echo "================================"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  all          - 모든 로그 (기본값)"
    echo "  errors       - 에러만 표시"
    echo "  warnings     - 경고만 표시"
    echo "  success      - 성공 메시지만 표시"
    echo "  images       - 이미지 처리 관련만 표시"
    echo "  sessions     - 세션 관련만 표시"
    echo "  api          - API 요청 관련만 표시"
    echo "  backend      - 백엔드 로그만"
    echo "  frontend     - 프론트엔드 로그만"
    echo "  help         - 이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0 errors    # 에러만 모니터링"
    echo "  $0 images    # 이미지 처리 과정만 추적"
    echo "  $0 sessions  # 세션 관리 추적"
}

# 로그 파일 경로 설정
BACKEND_LOG="server.log"
FRONTEND_LOG="frontend.log"
BACKEND_APP_LOG="backend/logs/app.log"

# 로그 파일 존재 확인
check_log_files() {
    echo -e "${BLUE}📋 로그 파일 상태 확인...${NC}"
    
    if [ -f "$BACKEND_LOG" ]; then
        echo -e "${GREEN}✅ 백엔드 서버 로그: $BACKEND_LOG${NC}"
    else
        echo -e "${YELLOW}⚠️ 백엔드 서버 로그 없음: $BACKEND_LOG${NC}"
    fi
    
    if [ -f "$FRONTEND_LOG" ]; then
        echo -e "${GREEN}✅ 프론트엔드 로그: $FRONTEND_LOG${NC}"
    else
        echo -e "${YELLOW}⚠️ 프론트엔드 로그 없음: $FRONTEND_LOG${NC}"
    fi
    
    if [ -f "$BACKEND_APP_LOG" ]; then
        echo -e "${GREEN}✅ 백엔드 앱 로그: $BACKEND_APP_LOG${NC}"
    else
        echo -e "${YELLOW}⚠️ 백엔드 앱 로그 없음: $BACKEND_APP_LOG${NC}"
    fi
    
    echo ""
}

# 실시간 모니터링 시작
start_monitoring() {
    local filter_type="$1"
    local log_source="$2"
    
    echo -e "${PURPLE}🔍 실시간 로그 모니터링 시작${NC}"
    echo -e "${CYAN}필터: $filter_type | 소스: $log_source${NC}"
    echo -e "${YELLOW}중지하려면 Ctrl+C를 누르세요${NC}"
    echo "=================================================================="
    
    case "$log_source" in
        "backend")
            if [ -f "$BACKEND_LOG" ]; then
                case "$filter_type" in
                    "errors") tail -f "$BACKEND_LOG" | filter_errors ;;
                    "warnings") tail -f "$BACKEND_LOG" | filter_warnings ;;
                    "success") tail -f "$BACKEND_LOG" | filter_success ;;
                    "images") tail -f "$BACKEND_LOG" | filter_image_processing ;;
                    "sessions") tail -f "$BACKEND_LOG" | filter_session ;;
                    "api") tail -f "$BACKEND_LOG" | filter_api ;;
                    *) tail -f "$BACKEND_LOG" ;;
                esac
            else
                echo -e "${RED}❌ 백엔드 로그 파일을 찾을 수 없습니다${NC}"
            fi
            ;;
        "frontend")
            if [ -f "$FRONTEND_LOG" ]; then
                case "$filter_type" in
                    "errors") tail -f "$FRONTEND_LOG" | filter_errors ;;
                    "warnings") tail -f "$FRONTEND_LOG" | filter_warnings ;;
                    "success") tail -f "$FRONTEND_LOG" | filter_success ;;
                    "images") tail -f "$FRONTEND_LOG" | filter_image_processing ;;
                    "sessions") tail -f "$FRONTEND_LOG" | filter_session ;;
                    "api") tail -f "$FRONTEND_LOG" | filter_api ;;
                    *) tail -f "$FRONTEND_LOG" ;;
                esac
            else
                echo -e "${RED}❌ 프론트엔드 로그 파일을 찾을 수 없습니다${NC}"
            fi
            ;;
        *)
            # 모든 로그 파일 동시 모니터링
            {
                if [ -f "$BACKEND_LOG" ]; then
                    tail -f "$BACKEND_LOG" | sed 's/^/[BACKEND] /' &
                fi
                
                if [ -f "$FRONTEND_LOG" ]; then
                    tail -f "$FRONTEND_LOG" | sed 's/^/[FRONTEND] /' &
                fi
                
                if [ -f "$BACKEND_APP_LOG" ]; then
                    tail -f "$BACKEND_APP_LOG" | sed 's/^/[APP] /' &
                fi
                
                wait
            } | case "$filter_type" in
                "errors") filter_errors ;;
                "warnings") filter_warnings ;;
                "success") filter_success ;;
                "images") filter_image_processing ;;
                "sessions") filter_session ;;
                "api") filter_api ;;
                *) cat ;;
            esac
            ;;
    esac
}

# 로그 요약 표시
show_log_summary() {
    echo -e "${PURPLE}📊 로그 요약 (최근 100줄 기준)${NC}"
    echo "=================================================="
    
    if [ -f "$BACKEND_LOG" ]; then
        echo -e "\n${BLUE}🔧 백엔드 서버 로그:${NC}"
        echo "에러 개수: $(tail -100 "$BACKEND_LOG" | grep -c "ERROR\|Failed" || echo 0)"
        echo "경고 개수: $(tail -100 "$BACKEND_LOG" | grep -c "WARNING\|WARN" || echo 0)"
        echo "성공 개수: $(tail -100 "$BACKEND_LOG" | grep -c "✅\|완료" || echo 0)"
        echo "이미지 처리: $(tail -100 "$BACKEND_LOG" | grep -c "이미지\|Step [1-8]" || echo 0)"
    fi
    
    if [ -f "$FRONTEND_LOG" ]; then
        echo -e "\n${CYAN}🎨 프론트엔드 로그:${NC}"
        echo "에러 개수: $(tail -100 "$FRONTEND_LOG" | grep -c "ERROR\|Failed" || echo 0)"
        echo "경고 개수: $(tail -100 "$FRONTEND_LOG" | grep -c "WARNING\|WARN" || echo 0)"
    fi
    
    echo ""
}

# 특정 세션 추적
track_session() {
    local session_id="$1"
    
    if [ -z "$session_id" ]; then
        echo -e "${RED}❌ 세션 ID를 입력해주세요${NC}"
        echo "사용법: $0 track-session <세션_ID>"
        return 1
    fi
    
    echo -e "${PURPLE}🔍 세션 추적: $session_id${NC}"
    echo "=================================================="
    
    # 모든 로그에서 해당 세션 ID 검색
    for log_file in "$BACKEND_LOG" "$FRONTEND_LOG" "$BACKEND_APP_LOG"; do
        if [ -f "$log_file" ]; then
            echo -e "\n${BLUE}📄 $(basename $log_file):${NC}"
            grep "$session_id" "$log_file" | tail -20 || echo "해당 세션 정보 없음"
        fi
    done
}

# 실시간 세션 추적
track_session_live() {
    local session_id="$1"
    
    if [ -z "$session_id" ]; then
        echo -e "${RED}❌ 세션 ID를 입력해주세요${NC}"
        echo "사용법: $0 track-session-live <세션_ID>"
        return 1
    fi
    
    echo -e "${PURPLE}🔍 실시간 세션 추적: $session_id${NC}"
    echo -e "${YELLOW}중지하려면 Ctrl+C를 누르세요${NC}"
    echo "=================================================="
    
    {
        if [ -f "$BACKEND_LOG" ]; then
            tail -f "$BACKEND_LOG" | grep --line-buffered "$session_id" | sed 's/^/[BACKEND] /' &
        fi
        
        if [ -f "$FRONTEND_LOG" ]; then
            tail -f "$FRONTEND_LOG" | grep --line-buffered "$session_id" | sed 's/^/[FRONTEND] /' &
        fi
        
        wait
    }
}

# 메인 실행 로직
case "$1" in
    "help"|"-h"|"--help")
        show_help
        ;;
    "summary")
        check_log_files
        show_log_summary
        ;;
    "track-session")
        track_session "$2"
        ;;
    "track-session-live")
        track_session_live "$2"
        ;;
    "backend")
        check_log_files
        start_monitoring "${2:-all}" "backend"
        ;;
    "frontend")
        check_log_files
        start_monitoring "${2:-all}" "frontend"
        ;;
    "errors"|"warnings"|"success"|"images"|"sessions"|"api")
        check_log_files
        start_monitoring "$1" "all"
        ;;
    *)
        check_log_files
        show_log_summary
        echo ""
        echo -e "${CYAN}실시간 모니터링을 시작하려면:${NC}"
        echo "  $0 all      # 모든 로그"
        echo "  $0 errors   # 에러만"
        echo "  $0 images   # 이미지 처리"
        echo "  $0 help     # 도움말"
        echo ""
        start_monitoring "all" "all"
        ;;
esac