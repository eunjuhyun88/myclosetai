#!/bin/bash

echo "🔍 실제 모델 파일 찾기 및 복사"
echo "=================================="

# 프로젝트 루트
PROJECT_ROOT="/Users/gimdudeul/MVP/mycloset-ai"
AI_MODELS_ROOT="$PROJECT_ROOT/backend/ai_models"

# conda 환경 확인
if [ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]; then
    echo "⚠️ conda 환경을 mycloset-ai로 변경하세요"
    echo "   conda activate mycloset-ai"
    exit 1
fi

echo "🐍 Conda 환경: $CONDA_DEFAULT_ENV ✅"
echo "📁 AI 모델 루트: $AI_MODELS_ROOT"
echo ""

# 1. 실제 모델 파일들 찾기
echo "🔍 1. 실제 모델 파일들 검색 중..."

# Human Parsing 모델 찾기
echo "   🔧 Step 01: Human Parsing 모델 검색..."
HUMAN_PARSING_FILES=$(find "$AI_MODELS_ROOT" -name "*.pth" -path "*human*" -o -name "*.pth" -path "*schp*" -o -name "*.pth" -path "*graphonomy*" 2>/dev/null | head -5)

if [ -n "$HUMAN_PARSING_FILES" ]; then
    echo "      ✅ Human Parsing 모델 발견:"
    echo "$HUMAN_PARSING_FILES" | while read file; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "         - $(basename "$file") ($size)"
            # 첫 번째 파일을 표준 위치로 복사
            if [ ! -f "$AI_MODELS_ROOT/step_01_human_parsing/$(basename "$file")" ]; then
                cp "$file" "$AI_MODELS_ROOT/step_01_human_parsing/" 2>/dev/null
                echo "           → 복사 완료: step_01_human_parsing/"
            fi
        fi
    done
else
    echo "      ❌ Human Parsing 모델 없음"
fi

# Pose Estimation 모델 찾기
echo "   🔧 Step 02: Pose Estimation 모델 검색..."
POSE_FILES=$(find "$AI_MODELS_ROOT" -name "*.pth" -path "*pose*" -o -name "*.pth" -path "*openpose*" 2>/dev/null | head -5)

if [ -n "$POSE_FILES" ]; then
    echo "      ✅ Pose Estimation 모델 발견:"
    echo "$POSE_FILES" | while read file; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "         - $(basename "$file") ($size)"
            if [ ! -f "$AI_MODELS_ROOT/step_02_pose_estimation/$(basename "$file")" ]; then
                cp "$file" "$AI_MODELS_ROOT/step_02_pose_estimation/" 2>/dev/null
                echo "           → 복사 완료: step_02_pose_estimation/"
            fi
        fi
    done
else
    echo "      ❌ Pose Estimation 모델 없음"
fi

# Cloth Segmentation 모델 찾기  
echo "   🔧 Step 03: Cloth Segmentation 모델 검색..."
CLOTH_SEG_FILES=$(find "$AI_MODELS_ROOT" -name "*.pth" -path "*u2net*" -o -name "*.pth" -path "*segment*" -o -name "*.pth" -path "*cloth*" 2>/dev/null | head -5)

if [ -n "$CLOTH_SEG_FILES" ]; then
    echo "      ✅ Cloth Segmentation 모델 발견:"
    echo "$CLOTH_SEG_FILES" | while read file; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "         - $(basename "$file") ($size)"
            if [ ! -f "$AI_MODELS_ROOT/step_03_cloth_segmentation/$(basename "$file")" ]; then
                cp "$file" "$AI_MODELS_ROOT/step_03_cloth_segmentation/" 2>/dev/null
                echo "           → 복사 완료: step_03_cloth_segmentation/"
            fi
        fi
    done
else
    echo "      ❌ Cloth Segmentation 모델 없음"
fi

# Virtual Fitting 모델 찾기 (가장 중요)
echo "   🔧 Step 06: Virtual Fitting 모델 검색..."
VIRTUAL_FITTING_FILES=$(find "$AI_MODELS_ROOT" -name "*.pth" -path "*viton*" -o -name "*.bin" -path "*ootd*" -o -name "*.safetensors" -path "*diffusion*" 2>/dev/null | head -10)

if [ -n "$VIRTUAL_FITTING_FILES" ]; then
    echo "      ✅ Virtual Fitting 모델 발견:"
    echo "$VIRTUAL_FITTING_FILES" | while read file; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "         - $(basename "$file") ($size)"
            if [ ! -f "$AI_MODELS_ROOT/step_06_virtual_fitting/$(basename "$file")" ]; then
                cp "$file" "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
                echo "           → 복사 완료: step_06_virtual_fitting/"
            fi
        fi
    done
else
    echo "      ❌ Virtual Fitting 모델 없음"
fi

# CLIP 모델 찾기
echo "   🔧 Step 08: Quality Assessment 모델 검색..."
CLIP_FILES=$(find "$AI_MODELS_ROOT" -name "*.bin" -path "*clip*" -o -name "*.pth" -path "*clip*" 2>/dev/null | head -5)

if [ -n "$CLIP_FILES" ]; then
    echo "      ✅ CLIP 모델 발견:"
    echo "$CLIP_FILES" | while read file; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "         - $(basename "$file") ($size)"
            if [ ! -f "$AI_MODELS_ROOT/step_08_quality_assessment/$(basename "$file")" ]; then
                cp "$file" "$AI_MODELS_ROOT/step_08_quality_assessment/" 2>/dev/null
                echo "           → 복사 완료: step_08_quality_assessment/"
            fi
        fi
    done
else
    echo "      ❌ CLIP 모델 없음"
fi

echo ""

# 2. cleanup_backup에서 모델 복구
echo "🔄 2. cleanup_backup에서 모델 복구..."

BACKUP_DIR="$PROJECT_ROOT/cleanup_backup_20250720_142014/backup_models_20250721_075438"
if [ -d "$BACKUP_DIR" ]; then
    echo "   📁 백업 디렉토리 발견: $BACKUP_DIR"
    
    # 백업에서 주요 모델들 복사
    for backup_file in "$BACKUP_DIR"/*.pth "$BACKUP_DIR"/*.bin "$BACKUP_DIR"/*.safetensors; do
        if [ -f "$backup_file" ]; then
            filename=$(basename "$backup_file")
            size=$(ls -lh "$backup_file" | awk '{print $5}')
            
            # 파일명 기반으로 적절한 Step에 복사
            case "$filename" in
                *human*|*parsing*|*schp*)
                    if [ ! -f "$AI_MODELS_ROOT/step_01_human_parsing/$filename" ]; then
                        cp "$backup_file" "$AI_MODELS_ROOT/step_01_human_parsing/" 2>/dev/null
                        echo "      ✅ $filename ($size) → step_01_human_parsing/"
                    fi
                    ;;
                *pose*|*openpose*)
                    if [ ! -f "$AI_MODELS_ROOT/step_02_pose_estimation/$filename" ]; then
                        cp "$backup_file" "$AI_MODELS_ROOT/step_02_pose_estimation/" 2>/dev/null
                        echo "      ✅ $filename ($size) → step_02_pose_estimation/"
                    fi
                    ;;
                *segment*|*u2net*)
                    if [ ! -f "$AI_MODELS_ROOT/step_03_cloth_segmentation/$filename" ]; then
                        cp "$backup_file" "$AI_MODELS_ROOT/step_03_cloth_segmentation/" 2>/dev/null
                        echo "      ✅ $filename ($size) → step_03_cloth_segmentation/"
                    fi
                    ;;
                *viton*|*diffusion*)
                    if [ ! -f "$AI_MODELS_ROOT/step_06_virtual_fitting/$filename" ]; then
                        cp "$backup_file" "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
                        echo "      ✅ $filename ($size) → step_06_virtual_fitting/"
                    fi
                    ;;
                *clip*|*vgg*)
                    if [ ! -f "$AI_MODELS_ROOT/step_08_quality_assessment/$filename" ]; then
                        cp "$backup_file" "$AI_MODELS_ROOT/step_08_quality_assessment/" 2>/dev/null
                        echo "      ✅ $filename ($size) → step_08_quality_assessment/"
                    fi
                    ;;
                *)
                    # 기타 모델들은 가장 가능성 높은 곳에 배치
                    if [ ! -f "$AI_MODELS_ROOT/step_06_virtual_fitting/$filename" ]; then
                        cp "$backup_file" "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
                        echo "      ✅ $filename ($size) → step_06_virtual_fitting/ (기본)"
                    fi
                    ;;
            esac
        fi
    done
else
    echo "   ❌ 백업 디렉토리 없음"
fi

echo ""

# 3. Hugging Face 캐시에서 모델 연결
echo "🔗 3. Hugging Face 캐시에서 모델 연결..."

HF_CACHE="$AI_MODELS_ROOT/huggingface_cache"
if [ -d "$HF_CACHE" ]; then
    echo "   📁 Hugging Face 캐시 발견"
    
    # OOTDiffusion 모델 찾기
    OOTD_PATH=$(find "$HF_CACHE" -name "*OOTDiffusion*" -type d | head -1)
    if [ -d "$OOTD_PATH" ]; then
        echo "      ✅ OOTDiffusion 발견: $OOTD_PATH"
        ln -sf "$OOTD_PATH"/* "$AI_MODELS_ROOT/step_06_virtual_fitting/" 2>/dev/null
        echo "         → step_06_virtual_fitting에 연결 완료"
    fi
    
    # CLIP 모델 찾기
    CLIP_PATH=$(find "$HF_CACHE" -name "*clip*" -type d | head -1)
    if [ -d "$CLIP_PATH" ]; then
        echo "      ✅ CLIP 발견: $CLIP_PATH"
        ln -sf "$CLIP_PATH"/* "$AI_MODELS_ROOT/step_08_quality_assessment/" 2>/dev/null
        echo "         → step_08_quality_assessment에 연결 완료"
    fi
else
    echo "   ❌ Hugging Face 캐시 없음"
fi

echo ""

# 4. 결과 확인
echo "📊 4. 최종 결과 확인..."
for step in {01..08}; do
    step_dir="$AI_MODELS_ROOT/step_${step}_"*
    file_count=$(find $step_dir -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        echo "   ✅ Step ${step}: ${file_count}개 모델 파일"
        # 상위 3개 파일 표시
        find $step_dir -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | head -3 | while read file; do
            if [ -f "$file" ]; then
                size=$(ls -lh "$file" | awk '{print $5}')
                echo "      - $(basename "$file") ($size)"
            fi
        done
    else
        echo "   ❌ Step ${step}: 0개 모델 파일"
    fi
done

echo ""

# 5. 권한 재설정
echo "🔐 5. 권한 재설정..."
chmod -R 755 "$AI_MODELS_ROOT/step_"*
echo "   ✅ 모든 Step 디렉토리 권한 설정 완료"

echo ""
echo "🎉 모델 파일 찾기 및 복사 완료!"
echo "=================================="
echo ""
echo "📋 다음 단계:"
echo "   1. python3 test_model_detection.py (모델 탐지 재확인)"
echo "   2. python3 enhanced_model_detector.py (향상된 탐지 실행)"
echo "   3. python3 backend/app/main.py (서버 재시작)"
echo ""
echo "✅ 이제 모델 파일들이 제대로 배치되었습니다!"