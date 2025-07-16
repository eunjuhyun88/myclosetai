# MyCloset AI - Backend

AI 기반 가상 피팅 플랫폼의 백엔드 서버

## 📁 프로젝트 구조

```
backend/
├── 📂 app/              # FastAPI 애플리케이션
├── 📂 ai_models/        # AI 모델 저장소
├── 📂 static/           # 정적 파일 (업로드, 결과)
├── 📂 scripts/          # 자동화 스크립트
├── 📂 configs/          # 설정 파일들
├── 📂 logs/             # 로그 파일들
└── 📄 run_server.py     # 서버 실행
```

## 🚀 빠른 시작

```bash
# 1. 가상환경 활성화
source mycloset_env/bin/activate  # 또는 source venv/bin/activate

# 2. 서버 실행
python run_server.py

# 3. API 문서 확인
# http://localhost:8000/docs
```

## 📋 스크립트

- `scripts/setup/` - 설치 및 설정
- `scripts/test/` - 테스트 스크립트
- `scripts/download/` - 모델 다운로드
- `scripts/utils/` - 유틸리티

## 🔧 설정

- `configs/requirements/` - Python 패키지 요구사항
- `configs/environment.yml` - Conda 환경 설정
