# MyCloset AI - 모델 검색 Makefile
# 편리한 모델 검색 및 관리 명령어들

.PHONY: help scan-models quick-scan deep-scan scan-pytorch scan-onnx scan-huggingface scan-local create-config clean-scan install-deps

# 기본 설정
PYTHON := python3
OUTPUT_DIR := ./discovered_models
CONFIG_DIR := ./configs
MIN_SIZE_MB := 1

# 색상 정의
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RED := \033[0;31m
NC := \033[0m

help: ## 사용 가능한 명령어 표시
	@echo "🔍 MyCloset AI 모델 검색 도구"
	@echo "=============================="
	@echo ""
	@echo "사용 가능한 명령어:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "예시:"
	@echo "  make quick-scan          # 빠른 검색"
	@echo "  make scan-pytorch        # PyTorch 모델만"
	@echo "  make deep-scan           # 전체 시스템 검색"
	@echo "  make create-config       # 설정 파일 생성"

# ==============================================
# 🔍 기본 검색 명령어들
# ==============================================

scan-models: ## 기본 모델 검색 (Python 스크립트)
	@echo "$(GREEN)🔍 AI 모델 기본 검색 시작...$(NC)"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) scan_models.py --output-dir $(OUTPUT_DIR)

quick-scan: ## 빠른 모델 검색 (Bash 스크립트)
	@echo "$(GREEN)⚡ 빠른 모델 검색 시작...$(NC)"
	@chmod +x quick_find_models.sh
	./quick_find_models.sh

deep-scan: ## 전체 시스템 딥 스캔 (시간 오래 걸림)
	@echo "$(YELLOW)⚠️  전체 시스템 스캔은 시간이 오래 걸릴 수 있습니다$(NC)"
	@read -p "계속하시겠습니까? [y/N]: " confirm && [[ $$confirm == [yY] ]] || exit 1
	@echo "$(GREEN)🔍 전체 시스템 딥 스캔 시작...$(NC)"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) scan_models.py --deep --output-dir $(OUTPUT_DIR)

# ==============================================
# 🎯 특정 모델 타입별 검색
# ==============================================

scan-pytorch: ## PyTorch 모델 검색 (.pth, .pt, .bin)
	@echo "$(GREEN)🔥 PyTorch 모델 검색...$(NC)"
	./quick_find_models.sh -t "pth,pt,bin" -o pytorch_models.txt

scan-onnx: ## ONNX 모델 검색
	@echo "$(GREEN)⚙️  ONNX 모델 검색...$(NC)"
	./quick_find_models.sh -t "onnx" -o onnx_models.txt

scan-tensorflow: ## TensorFlow 모델 검색
	@echo "$(GREEN)🟡 TensorFlow 모델 검색...$(NC)"
	./quick_find_models.sh -t "pb,h5,tflite" -o tensorflow_models.txt

scan-safetensors: ## SafeTensors 모델 검색
	@echo "$(GREEN)🔒 SafeTensors 모델 검색...$(NC)"
	./quick_find_models.sh -t "safetensors" -o safetensors_models.txt

# ==============================================
# 📁 특정 경로별 검색
# ==============================================

scan-huggingface: ## Hugging Face 캐시 검색
	@echo "$(GREEN)🤗 Hugging Face 캐시 검색...$(NC)"
	@if [ -d "$$HOME/.cache/huggingface" ]; then \
		./quick_find_models.sh -p "$$HOME/.cache/huggingface" -o huggingface_models.txt; \
	else \
		echo "$(YELLOW)⚠️  Hugging Face 캐시를 찾을 수 없습니다$(NC)"; \
	fi

scan-local: ## 현재 프로젝트 디렉토리만 검색
	@echo "$(GREEN)📂 로컬 프로젝트 검색...$(NC)"
	./quick_find_models.sh -p "." -o local_models.txt

scan-downloads: ## Downloads 폴더 검색
	@echo "$(GREEN)⬇️  Downloads 폴더 검색...$(NC)"
	@if [ -d "$$HOME/Downloads" ]; then \
		./quick_find_models.sh -p "$$HOME/Downloads" -o downloads_models.txt; \
	else \
		echo "$(YELLOW)⚠️  Downloads 폴더를 찾을 수 없습니다$(NC)"; \
	fi

scan-conda: ## Conda 환경 검색
	@echo "$(GREEN)🐍 Conda 환경 검색...$(NC)"
	@for env_path in "$$HOME/anaconda3/envs" "$$HOME/miniconda3/envs" "/opt/anaconda3/envs" "/opt/miniconda3/envs"; do \
		if [ -d "$$env_path" ]; then \
			echo "$(BLUE)검색 중: $$env_path$(NC)"; \
			./quick_find_models.sh -p "$$env_path" -o conda_models.txt; \
			break; \
		fi; \
	done

# ==============================================
# 🎯 MyCloset AI Step별 모델 검색
# ==============================================

scan-step-01: ## Step 01 Human Parsing 모델 검색
	@echo "$(GREEN)👤 Step 01 Human Parsing 모델 검색...$(NC)"
	@$(PYTHON) -c "import re; import glob; \
	files = []; \
	for ext in ['**/*.pth', '**/*.pt', '**/*.bin']: \
		files.extend(glob.glob(ext, recursive=True)); \
	human_parsing = [f for f in files if re.search(r'(human.*parsing|graphonomy|schp|atr)', f.lower())]; \
	print('\n'.join(human_parsing)) if human_parsing else print('Human Parsing 모델을 찾을 수 없습니다')"

scan-step-02: ## Step 02 Pose Estimation 모델 검색
	@echo "$(GREEN)🤸 Step 02 Pose Estimation 모델 검색...$(NC)"
	@$(PYTHON) -c "import re; import glob; \
	files = []; \
	for ext in ['**/*.pth', '**/*.pt', '**/*.bin']: \
		files.extend(glob.glob(ext, recursive=True)); \
	pose = [f for f in files if re.search(r'(pose.*estimation|openpose|mediapipe|pose.*net)', f.lower())]; \
	print('\n'.join(pose)) if pose else print('Pose Estimation 모델을 찾을 수 없습니다')"

scan-step-03: ## Step 03 Cloth Segmentation 모델 검색
	@echo "$(GREEN)👕 Step 03 Cloth Segmentation 모델 검색...$(NC)"
	@$(PYTHON) -c "import re; import glob; \
	files = []; \
	for ext in ['**/*.pth', '**/*.pt', '**/*.bin']: \
		files.extend(glob.glob(ext, recursive=True)); \
	cloth = [f for f in files if re.search(r'(cloth.*seg|u2net|sam|segment)', f.lower())]; \
	print('\n'.join(cloth)) if cloth else print('Cloth Segmentation 모델을 찾을 수 없습니다')"

scan-step-06: ## Step 06 Virtual Fitting 모델 검색
	@echo "$(GREEN)🎭 Step 06 Virtual Fitting 모델 검색...$(NC)"
	@$(PYTHON) -c "import re; import glob; \
	files = []; \
	for ext in ['**/*.pth', '**/*.pt', '**/*.bin', '**/*.safetensors']: \
		files.extend(glob.glob(ext, recursive=True)); \
	fitting = [f for f in files if re.search(r'(virtual.*fitting|ootdiffusion|stable.*diffusion|viton)', f.lower())]; \
	print('\n'.join(fitting)) if fitting else print('Virtual Fitting 모델을 찾을 수 없습니다')"

scan-all-steps: ## 모든 Step별 모델 검색
	@echo "$(GREEN)🎯 모든 Step 모델 검색...$(NC)"
	@make scan-step-01
	@echo ""
	@make scan-step-02
	@echo ""
	@make scan-step-03
	@echo ""
	@make scan-step-06

# ==============================================
# 📊 모델 통계 및 분석
# ==============================================

count-models: ## 모델 파일 개수만 표시
	@echo "$(GREEN)📊 모델 파일 개수 계산...$(NC)"
	@total=$(./quick_find_models.sh --count --quiet); \
	echo "총 $total개 모델 파일 발견"

size-report: ## 모델 크기별 분석 리포트
	@echo "$(GREEN)📊 모델 크기 분석...$(NC)"
	@echo "크기별 분류:"
	@echo "  소형 (1-10MB):"
	@./quick_find_models.sh -s 1 --quiet | wc -l | xargs -I {} echo "    {}개"
	@echo "  중형 (10-100MB):"
	@./quick_find_models.sh -s 10 --quiet | wc -l | xargs -I {} echo "    {}개"
	@echo "  대형 (100MB+):"
	@./quick_find_models.sh -s 100 --quiet | wc -l | xargs -I {} echo "    {}개"

framework-report: ## 프레임워크별 모델 분석
	@echo "$(GREEN)📊 프레임워크별 분석...$(NC)"
	@echo "PyTorch 모델:"
	@./quick_find_models.sh -t "pth,pt,bin" --count --quiet | xargs -I {} echo "  {}개"
	@echo "TensorFlow 모델:"
	@./quick_find_models.sh -t "pb,h5,tflite" --count --quiet | xargs -I {} echo "  {}개"
	@echo "ONNX 모델:"
	@./quick_find_models.sh -t "onnx" --count --quiet | xargs -I {} echo "  {}개"
	@echo "SafeTensors 모델:"
	@./quick_find_models.sh -t "safetensors" --count --quiet | xargs -I {} echo "  {}개"

# ==============================================
# ⚙️  설정 파일 생성 및 관리
# ==============================================

create-config: ## 설정 파일들 자동 생성
	@echo "$(GREEN)📝 설정 파일 생성...$(NC)"
	@mkdir -p $(CONFIG_DIR)
	$(PYTHON) scan_models.py --create-config --output-dir $(CONFIG_DIR)
	@echo "$(GREEN)✅ 설정 파일 생성 완료: $(CONFIG_DIR)$(NC)"

update-config: ## 기존 설정 파일 업데이트
	@echo "$(GREEN)🔄 설정 파일 업데이트...$(NC)"
	@if [ -d "$(CONFIG_DIR)" ]; then \
		rm -rf $(CONFIG_DIR)/*.json $(CONFIG_DIR)/*.yaml $(CONFIG_DIR)/*.py $(CONFIG_DIR)/*.sh; \
	fi
	@make create-config

validate-config: ## 생성된 설정 파일 검증
	@echo "$(GREEN)✅ 설정 파일 검증...$(NC)"
	@if [ -f "$(CONFIG_DIR)/model_paths.py" ]; then \
		echo "$(GREEN)✅ Python 설정 파일 존재$(NC)"; \
		$(PYTHON) -c "import sys; sys.path.append('$(CONFIG_DIR)'); import model_paths; model_paths.print_scan_summary()"; \
	else \
		echo "$(RED)❌ Python 설정 파일 없음$(NC)"; \
	fi
	@if [ -f "$(CONFIG_DIR)/models_config.yaml" ]; then \
		echo "$(GREEN)✅ YAML 설정 파일 존재$(NC)"; \
	else \
		echo "$(RED)❌ YAML 설정 파일 없음$(NC)"; \
	fi

# ==============================================
# 🧹 정리 및 유지보수
# ==============================================

clean-scan: ## 검색 결과 파일들 정리
	@echo "$(YELLOW)🧹 검색 결과 파일 정리...$(NC)"
	@rm -f *_models.txt
	@rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)✅ 정리 완료$(NC)"

clean-config: ## 설정 파일들 정리
	@echo "$(YELLOW)🧹 설정 파일 정리...$(NC)"
	@rm -rf $(CONFIG_DIR)
	@echo "$(GREEN)✅ 설정 파일 정리 완료$(NC)"

clean-all: clean-scan clean-config ## 모든 생성된 파일 정리
	@echo "$(GREEN)✅ 전체 정리 완료$(NC)"

# ==============================================
# 📦 의존성 및 설치
# ==============================================

install-deps: ## 필요한 Python 패키지 설치
	@echo "$(GREEN)📦 의존성 패키지 설치...$(NC)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install tqdm pyyaml
	@if command -v conda >/dev/null 2>&1; then \
		echo "$(BLUE)Conda 환경에서 추가 패키지 설치...$(NC)"; \
		conda install -c conda-forge tqdm pyyaml; \
	fi
	@echo "$(GREEN)✅ 의존성 설치 완료$(NC)"

check-deps: ## 의존성 확인
	@echo "$(GREEN)🔍 의존성 확인...$(NC)"
	@echo "Python 버전:"
	@$(PYTHON) --version
	@echo ""
	@echo "필수 패키지 확인:"
	@$(PYTHON) -c "import tqdm; print('✅ tqdm 설치됨')" 2>/dev/null || echo "❌ tqdm 없음"
	@$(PYTHON) -c "import yaml; print('✅ PyYAML 설치됨')" 2>/dev/null || echo "❌ PyYAML 없음"
	@$(PYTHON) -c "import torch; print('✅ PyTorch 설치됨')" 2>/dev/null || echo "❌ PyTorch 없음"
	@echo ""
	@echo "시스템 도구:"
	@command -v find >/dev/null && echo "✅ find 명령어 사용 가능" || echo "❌ find 명령어 없음"
	@command -v bc >/dev/null && echo "✅ bc 계산기 사용 가능" || echo "❌ bc 계산기 없음"

# ==============================================
# 🚀 원클릭 스크립트들
# ==============================================

quick-setup: install-deps ## 빠른 설정 (의존성 설치 + 빠른 스캔)
	@echo "$(GREEN)🚀 빠른 설정 시작...$(NC)"
	@chmod +x quick_find_models.sh
	@make quick-scan

full-setup: install-deps create-config ## 완전 설정 (의존성 + 스캔 + 설정)
	@echo "$(GREEN)🚀 완전 설정 시작...$(NC)"
	@chmod +x quick_find_models.sh
	@make scan-models

mycloset-scan: ## MyCloset AI 전용 모델 검색
	@echo "$(GREEN)🎯 MyCloset AI 모델 전용 검색...$(NC)"
	@make scan-all-steps
	@echo ""
	@make count-models
	@echo ""
	@echo "$(BLUE)📁 추천 모델 다운로드 경로:$(NC)"
	@echo "  mkdir -p ./ai_models/checkpoints/step_01_human_parsing"
	@echo "  mkdir -p ./ai_models/checkpoints/step_02_pose_estimation"
	@echo "  mkdir -p ./ai_models/checkpoints/step_03_cloth_segmentation"
	@echo "  mkdir -p ./ai_models/checkpoints/step_06_virtual_fitting"

# ==============================================
# 📋 도움말 및 정보
# ==============================================

show-commands: ## 자주 사용하는 명령어 표시
	@echo "$(GREEN)📋 자주 사용하는 명령어들:$(NC)"
	@echo ""
	@echo "$(BLUE)빠른 검색:$(NC)"
	@echo "  make quick-scan              # 기본 빠른 검색"
	@echo "  make scan-pytorch            # PyTorch 모델만"
	@echo "  make mycloset-scan           # MyCloset AI 전용"
	@echo ""
	@echo "$(BLUE)설정 관리:$(NC)"
	@echo "  make create-config           # 설정 파일 생성"
	@echo "  make validate-config         # 설정 파일 확인"
	@echo ""
	@echo "$(BLUE)정리:$(NC)"
	@echo "  make clean-scan              # 검색 결과 정리"
	@echo "  make clean-all               # 모든 파일 정리"

show-paths: ## 검색 대상 경로들 표시
	@echo "$(GREEN)📂 검색 대상 경로들:$(NC)"
	@echo ""
	@echo "$(BLUE)프로젝트 경로:$(NC)"
	@echo "  ./ai_models"
	@echo "  ./models"
	@echo "  ./checkpoints"
	@echo ""
	@echo "$(BLUE)시스템 캐시:$(NC)"
	@echo "  $HOME/.cache/huggingface"
	@echo "  $HOME/.cache/torch"
	@echo "  $HOME/.cache/transformers"
	@echo ""
	@echo "$(BLUE)Conda 환경:$(NC)"
	@echo "  $HOME/anaconda3/envs"
	@echo "  $HOME/miniconda3/envs"

# ==============================================
# 🔧 고급 기능
# ==============================================

find-duplicates: ## 중복 모델 파일 찾기 (체크섬 기반)
	@echo "$(GREEN)🔍 중복 모델 파일 검색...$(NC)"
	@$(PYTHON) scan_models.py --output-dir $(OUTPUT_DIR) --create-config >/dev/null 2>&1
	@if [ -f "$(OUTPUT_DIR)/discovered_models.json" ]; then \
		$(PYTHON) -c "import json; \
		with open('$(OUTPUT_DIR)/discovered_models.json') as f: data=json.load(f); \
		checksums={}; \
		for model in data['models']: \
			checksum = model['checksum']; \
			if checksum in checksums: checksums[checksum].append(model['path']); \
			else: checksums[checksum] = [model['path']]; \
		duplicates = {k:v for k,v in checksums.items() if len(v)>1}; \
		print('중복 파일:', len(duplicates), '그룹') if duplicates else print('중복 파일 없음'); \
		[print(f'  체크섬 {k[:8]}...: {len(v)}개 파일') for k,v in duplicates.items()]"; \
	else \
		echo "$(RED)❌ 먼저 make scan-models를 실행하세요$(NC)"; \
	fi

find-large-models: ## 큰 모델 파일들 찾기 (100MB+)
	@echo "$(GREEN)🐘 대용량 모델 파일 검색 (100MB+)...$(NC)"
	@./quick_find_models.sh -s 100

optimize-storage: ## 스토리지 최적화 제안
	@echo "$(GREEN)💾 스토리지 최적화 제안...$(NC)"
	@echo ""
	@make find-duplicates
	@echo ""
	@make find-large-models
	@echo ""
	@echo "$(BLUE)💡 최적화 팁:$(NC)"
	@echo "  1. 중복 파일 제거"
	@echo "  2. 사용하지 않는 모델 삭제" 
	@echo "  3. 모델을 symlink로 공유"
	@echo "  4. 압축 저장 고려"

# ==============================================
# 🏃 원클릭 실행 스크립트
# ==============================================

demo: ## 데모 실행 (모든 기능 시연)
	@echo "$(GREEN)🎬 MyCloset AI 모델 스캐너 데모$(NC)"
	@echo "======================================"
	@echo ""
	@make check-deps
	@echo ""
	@make quick-scan
	@echo ""
	@make count-models
	@echo ""
	@make framework-report
	@echo ""
	@echo "$(GREEN)🎉 데모 완료! 더 많은 명령어는 'make help' 참조$(NC)"

# 기본 타겟을 help로 설정
.DEFAULT_GOAL := help