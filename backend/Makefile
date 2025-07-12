# backend/Makefile
# MyCloset AI Backend 개발 명령어

.PHONY: help setup install run dev test clean lint format docker-build docker-run benchmark

# 기본 설정
PYTHON := python3
PIP := pip
VENV := venv
PROJECT_NAME := mycloset-ai-backend

# 색상 코드
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## 사용 가능한 명령어 표시
	@echo "$(GREEN)MyCloset AI Backend 개발 명령어:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)예제:$(NC)"
	@echo "  make setup     # 초기 환경 설정"
	@echo "  make run       # 개발 서버 실행"
	@echo "  make test      # 테스트 실행"

setup: ## 초기 환경 설정 (가상환경 생성)
	@echo "$(GREEN)🔧 초기 환경 설정 시작...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✅ 가상환경 생성 완료$(NC)"
	@echo "$(YELLOW)다음 명령어를 실행하세요: source $(VENV)/bin/activate$(NC)"

install: ## 의존성 설치 (M3 Max 최적화)
	@echo "$(GREEN)📦 M3 Max 최적화 패키지 설치 중...$(NC)"
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-mac.txt
	@echo "$(GREEN)✅ 의존성 설치 완료$(NC)"

install-dev: ## 개발용 의존성 추가 설치
	@echo "$(GREEN)🛠️ 개발용 패키지 설치 중...$(NC)"
	$(VENV)/bin/pip install pytest pytest-asyncio pytest-cov black flake8 mypy
	@echo "$(GREEN)✅ 개발용 패키지 설치 완료$(NC)"

run: ## 개발 서버 실행
	@echo "$(GREEN)🚀 MyCloset AI 백엔드 서버 시작...$(NC)"
	@echo "$(YELLOW)서버 주소: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API 문서: http://localhost:8000/docs$(NC)"
	@echo "$(YELLOW)헬스체크: http://localhost:8000/health$(NC)"
	@echo ""
	$(VENV)/bin/python run_server.py

dev: ## 개발 모드 (자동 재시작)
	@echo "$(GREEN)🔄 개발 모드로 서버 시작 (자동 재시작)...$(NC)"
	$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

prod: ## 프로덕션 모드
	@echo "$(GREEN)🏭 프로덕션 모드로 서버 시작...$(NC)"
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

test: ## 테스트 실행
	@echo "$(GREEN)🧪 테스트 실행 중...$(NC)"
	$(VENV)/bin/python -m pytest tests/ -v --cov=app --cov-report=html
	@echo "$(GREEN)✅ 테스트 완료$(NC)"
	@echo "$(YELLOW)커버리지 리포트: htmlcov/index.html$(NC)"

test-fast: ## 빠른 테스트 (커버리지 제외)
	@echo "$(GREEN)⚡ 빠른 테스트 실행...$(NC)"
	$(VENV)/bin/python -m pytest tests/ -v -x

benchmark: ## GPU 벤치마크 실행
	@echo "$(GREEN)📊 M3 Max GPU 벤치마크 실행...$(NC)"
	$(VENV)/bin/python -c "from app.core.gpu_config import run_benchmark; print(run_benchmark())"

lint: ## 코드 린팅 검사
	@echo "$(GREEN)🔍 코드 린팅 검사...$(NC)"
	$(VENV)/bin/flake8 app/ --max-line-length=100 --ignore=E203,W503
	$(VENV)/bin/mypy app/ --ignore-missing-imports
	@echo "$(GREEN)✅ 린팅 검사 완료$(NC)"

format: ## 코드 포맷팅
	@echo "$(GREEN)✨ 코드 포맷팅...$(NC)"
	$(VENV)/bin/black app/ tests/ --line-length=100
	@echo "$(GREEN)✅ 포맷팅 완료$(NC)"

clean: ## 캐시 및 임시 파일 정리
	@echo "$(GREEN)🧹 정리 작업 시작...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .coverage .mypy_cache/
	rm -rf static/uploads/* static/results/* static/temp/* 2>/dev/null || true
	@echo "$(GREEN)✅ 정리 완료$(NC)"

clean-models: ## AI 모델 캐시 정리
	@echo "$(GREEN)🤖 AI 모델 캐시 정리...$(NC)"
	rm -rf ai_models/cache/* ai_models/temp/* 2>/dev/null || true
	@echo "$(GREEN)✅ 모델 캐시 정리 완료$(NC)"

logs: ## 로그 확인
	@echo "$(GREEN)📋 최근 로그:$(NC)"
	@if [ -f logs/app.log ]; then \
		tail -n 50 logs/app.log; \
	else \
		echo "$(YELLOW)로그 파일이 없습니다.$(NC)"; \
	fi

status: ## 서버 상태 확인
	@echo "$(GREEN)📊 서버 상태 확인...$(NC)"
	@curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "$(RED)서버가 실행되지 않았습니다$(NC)"

api-test: ## API 기본 테스트
	@echo "$(GREEN)🔌 API 연결 테스트...$(NC)"
	@curl -s http://localhost:8000/health || echo "$(RED)❌ 백엔드 서버 연결 실패$(NC)"
	@curl -s http://localhost:8000/api/status || echo "$(RED)❌ API 상태 확인 실패$(NC)"
	@echo "$(GREEN)✅ API 테스트 완료$(NC)"

docker-build: ## Docker 이미지 빌드
	@echo "$(GREEN)🐳 Docker 이미지 빌드...$(NC)"
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✅ Docker 이미지 빌드 완료$(NC)"

docker-run: ## Docker 컨테이너 실행
	@echo "$(GREEN)🐳 Docker 컨테이너 실행...$(NC)"
	docker run -p 8000:8000 -v $(PWD)/static:/app/static $(PROJECT_NAME):latest

docker-stop: ## Docker 컨테이너 중지
	@echo "$(GREEN)🛑 Docker 컨테이너 중지...$(NC)"
	docker ps -q --filter ancestor=$(PROJECT_NAME) | xargs -r docker stop

install-all: setup install install-dev ## 전체 설치 (setup + install + dev)
	@echo "$(GREEN)🎉 전체 설치 완료!$(NC)"
	@echo "$(YELLOW)이제 'make run' 또는 'make dev'로 서버를 시작하세요$(NC)"

reset: clean setup install ## 환경 초기화 후 재설치
	@echo "$(GREEN)🔄 환경 초기화 및 재설치 완료$(NC)"

check-deps: ## 의존성 확인
	@echo "$(GREEN)📋 의존성 확인...$(NC)"
	@$(VENV)/bin/pip list | grep -E "(torch|fastapi|pillow|opencv)" || echo "$(YELLOW)일부 패키지가 설치되지 않았습니다$(NC)"

info: ## 환경 정보 표시
	@echo "$(GREEN)ℹ️ 환경 정보:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "가상환경: $(VENV)"
	@echo "프로젝트: $(PROJECT_NAME)"
	@echo "플랫폼: $(shell uname -sm)"
	@if [ -d "$(VENV)" ]; then \
		echo "가상환경 상태: ✅ 생성됨"; \
		if [ -f "$(VENV)/bin/python" ]; then \
			echo "Python 경로: $(VENV)/bin/python"; \
		fi; \
	else \
		echo "가상환경 상태: ❌ 없음"; \
	fi

# 개발 워크플로우
dev-setup: setup install install-dev format ## 개발 환경 완전 설정
	@echo "$(GREEN)🎯 개발 환경 설정 완료!$(NC)"
	@echo "$(YELLOW)다음 단계:$(NC)"
	@echo "1. source $(VENV)/bin/activate"
	@echo "2. make run"

# 배포 준비
deploy-check: test lint ## 배포 전 체크
	@echo "$(GREEN)✅ 배포 준비 체크 완료$(NC)"

# 성능 모니터링
monitor: ## 실시간 로그 모니터링
	@echo "$(GREEN)👀 실시간 로그 모니터링 (Ctrl+C로 종료)$(NC)"
	@tail -f logs/app.log 2>/dev/null || echo "$(YELLOW)로그 파일 생성 대기 중...$(NC)"