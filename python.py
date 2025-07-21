#!/bin/bash

# 🔥 MyCloset AI Conda 환경 영구 설정 스크립트
# 한 번 실행하면 이후에는 자동으로 PYTHONPATH가 설정됩니다

echo "🔧 Conda 환경 영구 설정 중..."

# conda 환경 활성화 스크립트에 PYTHONPATH 추가
CONDA_ENV_PATH=$(conda info --base)/envs/mycloset-ai

# activate.d 디렉토리 생성
mkdir -p $CONDA_ENV_PATH/etc/conda/activate.d
mkdir -p $CONDA_ENV_PATH/etc/conda/deactivate.d

# PYTHONPATH 설정 스크립트 생성
cat > $CONDA_ENV_PATH/etc/conda/activate.d/mycloset_ai_env.sh << 'EOF'
#!/bin/bash
# MyCloset AI 프로젝트 PYTHONPATH 설정
export MYCLOSET_AI_ROOT="/Users/gimdudeul/MVP/mycloset-ai/backend"
export PYTHONPATH="$MYCLOSET_AI_ROOT:$PYTHONPATH"
echo "✅ MyCloset AI PYTHONPATH 설정 완료: $MYCLOSET_AI_ROOT"
EOF

# PYTHONPATH 해제 스크립트 생성
cat > $CONDA_ENV_PATH/etc/conda/deactivate.d/mycloset_ai_env.sh << 'EOF'
#!/bin/bash
# MyCloset AI 프로젝트 PYTHONPATH 해제
unset MYCLOSET_AI_ROOT
echo "🔄 MyCloset AI PYTHONPATH 해제 완료"
EOF

# 실행 권한 부여
chmod +x $CONDA_ENV_PATH/etc/conda/activate.d/mycloset_ai_env.sh
chmod +x $CONDA_ENV_PATH/etc/conda/deactivate.d/mycloset_ai_env.sh

echo "✅ Conda 환경 영구 설정 완료!"
echo "📝 이제 'conda activate mycloset-ai' 할 때마다 자동으로 PYTHONPATH가 설정됩니다"

# 현재 세션에서도 적용
export MYCLOSET_AI_ROOT="/Users/gimdudeul/MVP/mycloset-ai/backend"
export PYTHONPATH="$MYCLOSET_AI_ROOT:$PYTHONPATH"

echo "🧪 설정 테스트..."
cd /Users/gimdudeul/MVP/mycloset-ai/backend

python -c "
try:
    from app.ai_pipeline.utils.auto_model_detector import quick_model_detection
    print('✅ Import 테스트 성공!')
except Exception as e:
    print(f'❌ Import 테스트 실패: {e}')
"

echo "🎉 모든 설정 완료!"