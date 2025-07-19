# 👗 MyCloset AI - AI 가상 피팅 플랫폼

AI 기술을 활용한 스마트 가상 피팅 서비스

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Conda 환경 생성 및 활성화
bash scripts/setup/setup_conda_env.sh
conda activate mycloset-m3

# 의존성 설치
cd backend && pip install -r requirements.txt
cd ../frontend && npm install
```

### 2. 개발 서버 실행
```bash
# 백엔드 (터미널 1)
cd backend && python app/main.py

# 프론트엔드 (터미널 2)  
cd frontend && npm run dev
```

## 📁 프로젝트 구조

```
mycloset-ai/
├── backend/           # FastAPI 백엔드
│   ├── app/          # 메인 애플리케이션
│   │   ├── ai_pipeline/  # AI 파이프라인 (8단계)
│   │   ├── api/      # API 라우터
│   │   ├── core/     # 핵심 설정
│   │   ├── models/   # 데이터 모델
│   │   └── services/ # 비즈니스 로직
│   ├── ai_models/    # AI 모델 파일들
│   └── static/       # 정적 파일
├── frontend/         # React 프론트엔드
│   ├── src/
│   │   ├── components/  # UI 컴포넌트
│   │   ├── pages/    # 페이지
│   │   └── hooks/    # 커스텀 훅
├── scripts/          # 유틸리티 스크립트
│   ├── setup/        # 환경 설정
│   ├── models/       # 모델 관리
│   ├── dev/          # 개발 도구
│   └── maintenance/  # 유지보수
├── logs/             # 로그 파일
├── reports/          # 분석 리포트
└── data/             # 데이터베이스 파일
```

## 🤖 AI 파이프라인 (8단계)

1. **Human Parsing** - 인체 부위 분석
2. **Pose Estimation** - 자세 추정  
3. **Cloth Segmentation** - 의류 분할
4. **Geometric Matching** - 기하학적 매칭
5. **Cloth Warping** - 의류 변형
6. **Virtual Fitting** - 가상 피팅
7. **Post Processing** - 후처리
8. **Quality Assessment** - 품질 평가

## 🛠️ 개발 도구

```bash
# 프로젝트 상태 체크
bash scripts/dev/check_structure.sh

# 모델 스캔
python scripts/models/complete_scanner.py

# 로그 모니터링  
bash scripts/dev/log_monitoring_script.sh
```

## 📋 요구사항

- Python 3.9+
- Node.js 18+
- macOS (M1/M2/M3 권장)
- 16GB+ RAM
- 10GB+ 저장공간

## 🔧 문제 해결

문제가 발생하면 다음을 확인하세요:

1. conda 환경 활성화: `conda activate mycloset-m3`
2. 모델 파일 확인: `ls backend/ai_models/`
3. 로그 확인: `tail -f logs/*.log`

## 📞 지원

- 이슈: GitHub Issues
- 문서: `/docs` 폴더 참고
