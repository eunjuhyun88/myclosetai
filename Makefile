.PHONY: help setup install run-backend run-frontend dev clean test

help:
	@echo "MyCloset AI 개발 명령어:"
	@echo "  setup        - 초기 환경 설정"
	@echo "  install      - 의존성 설치" 
	@echo "  run-backend  - 백엔드 서버 실행"
	@echo "  run-frontend - 프론트엔드 서버 실행"
	@echo "  dev          - 개발 환경 전체 설정"
	@echo "  test         - 서버 연결 테스트"
	@echo "  clean        - 정리"

setup:
	@echo "🔧 초기 환경 설정 중..."
	cd backend && python3 -m venv venv
	cd frontend && npm install

install:
	@echo "📦 의존성 설치 중..."
	cd backend && source venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

run-backend:
	@echo "🔧 백엔드 서버 시작 중..."
	cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	@echo "🎨 프론트엔드 서버 시작 중..."
	cd frontend && npm run dev

# 백엔드 직접 실행 (python으로)
run-backend-direct:
	@echo "🔧 백엔드 직접 실행 중..."
	cd backend && source venv/bin/activate && python app/main.py

dev: install
	@echo "🚀 개발 서버들을 시작합니다..."
	@echo "백엔드는 http://localhost:8000"
	@echo "프론트엔드는 http://localhost:5173"

test:
	@echo "🔍 서버 연결 테스트 중..."
	@curl -s http://localhost:8000/health || echo "❌ 백엔드 서버가 실행되지 않았습니다."

clean:
	@echo "🧹 정리 중..."
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
