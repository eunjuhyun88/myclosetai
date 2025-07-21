#!/bin/bash

echo "📦 실제 모델 파일 이동 스크립트"
echo "=================================="
echo "⚠️ 심볼릭 링크 대신 실제 파일을 복사/이동합니다"

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

# 1. 기존 심볼릭 링크 정리
echo "🧹 1. 기존 심볼릭 링크 및 잘못된 파일 정리..."
for step in {01..08}; do
    step_dir="$AI_MODELS_ROOT/step_${step}_"*
    if [ -d $step_dir ]; then
        echo "   정리 중: $(basename $step_dir)"
        # 심볼릭 링크 제거
        find $step_dir -type l -delete 2>/dev/null
        # 빈 디렉토리 제거
        find $step_dir -type d -empty -delete 2>/dev/null
        echo "      ✅ 심볼릭 링크 정리 완료"
    fi
done

echo ""

# 2. 실제 모델 파일들 찾아서 복사
echo "📋 2. 실제 모델 파일 검색 및 복사..."

# Step 01: Human Parsing
echo "   🔧 Step 01: Human Parsing"
STEP01_DIR="$AI_MODELS_ROOT/step_01_human_parsing"
mkdir -p "$STEP01_DIR"

# Human Parsing 관련 파일들 찾기
find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(human|parsing|schp|graphonomy|atr|lip)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP01_DIR/$filename" ]; then
            cp "$file" "$STEP01_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 02: Pose Estimation  
echo "   🔧 Step 02: Pose Estimation"
STEP02_DIR="$AI_MODELS_ROOT/step_02_pose_estimation"
mkdir -p "$STEP02_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(pose|openpose|body|hrnet|coco)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP02_DIR/$filename" ]; then
            cp "$file" "$STEP02_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 03: Cloth Segmentation
echo "   🔧 Step 03: Cloth Segmentation"
STEP03_DIR="$AI_MODELS_ROOT/step_03_cloth_segmentation"
mkdir -p "$STEP03_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(u2net|segment|cloth|rembg|sam)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP03_DIR/$filename" ]; then
            cp "$file" "$STEP03_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 04: Geometric Matching
echo "   🔧 Step 04: Geometric Matching"
STEP04_DIR="$AI_MODELS_ROOT/step_04_geometric_matching"
mkdir -p "$STEP04_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(gmm|geometric|matching|tps|warp)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP04_DIR/$filename" ]; then
            cp "$file" "$STEP04_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 05: Cloth Warping
echo "   🔧 Step 05: Cloth Warping"
STEP05_DIR="$AI_MODELS_ROOT/step_05_cloth_warping"
mkdir -p "$STEP05_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(tom|warp|warping|tps)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP05_DIR/$filename" ]; then
            cp "$file" "$STEP05_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 06: Virtual Fitting (가장 중요)
echo "   🔧 Step 06: Virtual Fitting"
STEP06_DIR="$AI_MODELS_ROOT/step_06_virtual_fitting"
mkdir -p "$STEP06_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) | grep -iE "(viton|ootd|diffusion|stable|unet|fitting)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP06_DIR/$filename" ]; then
            cp "$file" "$STEP06_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 07: Post Processing
echo "   🔧 Step 07: Post Processing"
STEP07_DIR="$AI_MODELS_ROOT/step_07_post_processing"
mkdir -p "$STEP07_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(enhance|super|resolution|upscale|esrgan)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP07_DIR/$filename" ]; then
            cp "$file" "$STEP07_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

# Step 08: Quality Assessment
echo "   🔧 Step 08: Quality Assessment"
STEP08_DIR="$AI_MODELS_ROOT/step_08_quality_assessment"
mkdir -p "$STEP08_DIR"

find "$AI_MODELS_ROOT" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" \) | grep -iE "(clip|quality|aesthetic|vgg|resnet)" | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP08_DIR/$filename" ]; then
            cp "$file" "$STEP08_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "      ✅ $filename ($size) 복사됨"
        fi
    fi
done

echo ""

# 3. 큰 파일들 (1GB 이상) 을 Virtual Fitting에 추가 배치
echo "🎯 3. 대용량 모델 파일 추가 배치..."
find "$AI_MODELS_ROOT" -type f -size +1G \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) | while read file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ ! -f "$STEP06_DIR/$filename" ]; then
            cp "$file" "$STEP06_DIR/" 2>/dev/null
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "   ✅ 대용량 파일: $filename ($size) → step_06_virtual_fitting"
        fi
    fi
done

echo ""

# 4. 백업 디렉토리에서 추가 복구
echo "🔄 4. 백업 디렉토리에서 추가 복구..."
BACKUP_DIRS=(
    "$PROJECT_ROOT/cleanup_backup_20250720_142014/backup_models_20250721_075438"
    "$PROJECT_ROOT/cleanup_backup_20250720_142014/backup_models_20250721_075044"
)

for BACKUP_DIR in "${BACKUP_DIRS[@]}"; do
    if [ -d "$BACKUP_DIR" ]; then
        echo "   📁 백업에서 복구: $(basename "$BACKUP_DIR")"
        
        find "$BACKUP_DIR" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) | while read backup_file; do
            if [ -f "$backup_file" ]; then
                filename=$(basename "$backup_file")
                size=$(ls -lh "$backup_file" | awk '{print $5}')
                
                # 파일명 기반으로 적절한 Step에 배치
                case "$filename" in
                    *human*|*parsing*|*schp*|*graphonomy*)
                        target_dir="$STEP01_DIR"
                        step_name="step_01_human_parsing"
                        ;;
                    *pose*|*openpose*|*body*)
                        target_dir="$STEP02_DIR"
                        step_name="step_02_pose_estimation"
                        ;;
                    *segment*|*u2net*|*cloth*)
                        target_dir="$STEP03_DIR"
                        step_name="step_03_cloth_segmentation"
                        ;;
                    *viton*|*diffusion*|*stable*|*ootd*)
                        target_dir="$STEP06_DIR"
                        step_name="step_06_virtual_fitting"
                        ;;
                    *clip*|*vgg*|*resnet*)
                        target_dir="$STEP08_DIR"
                        step_name="step_08_quality_assessment"
                        ;;
                    *)
                        target_dir="$STEP06_DIR"  # 기본적으로 Virtual Fitting에
                        step_name="step_06_virtual_fitting"
                        ;;
                esac
                
                if [ ! -f "$target_dir/$filename" ]; then
                    cp "$backup_file" "$target_dir/" 2>/dev/null
                    echo "      ✅ $filename ($size) → $step_name"
                fi
            fi
        done
    fi
done

echo ""

# 5. 권한 설정
echo "🔐 5. 권한 설정..."
chmod -R 755 "$AI_MODELS_ROOT/step_"*
echo "   ✅ 모든 Step 디렉토리 권한 설정 완료"

echo ""

# 6. 최종 결과 확인
echo "📊 6. 최종 결과 확인..."
total_files=0
total_size_gb=0

for step in {01..08}; do
    step_dir=$(find "$AI_MODELS_ROOT" -name "step_${step}_*" -type d | head -1)
    if [ -d "$step_dir" ]; then
        file_count=$(find "$step_dir" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null | wc -l)
        
        if [ "$file_count" -gt 0 ]; then
            # 크기 계산
            size_bytes=$(find "$step_dir" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) -exec stat -f%z {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
            if [ -n "$size_bytes" ] && [ "$size_bytes" -gt 0 ]; then
                size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0")
            else
                size_gb="0"
            fi
            
            echo "   ✅ Step ${step}: ${file_count}개 모델 파일 (${size_gb}GB)"
            
            # 상위 3개 파일 표시
            find "$step_dir" -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" \) -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -3 | while read line; do
                filename=$(echo "$line" | awk '{print $9}' | xargs basename)
                size=$(echo "$line" | awk '{print $5}')
                echo "      - $filename ($size)"
            done
            
            total_files=$((total_files + file_count))
            total_size_gb=$(echo "$total_size_gb + $size_gb" | bc -l 2>/dev/null || echo "$total_size_gb")
        else
            echo "   ❌ Step ${step}: 0개 모델 파일"
        fi
    else
        echo "   ❌ Step ${step}: 디렉토리 없음"
    fi
done

echo ""
echo "="*60
echo "📊 전체 요약"
echo "="*60
echo "🔍 총 ${total_files}개 모델 파일 복사됨"
echo "💾 총 크기: ${total_size_gb}GB"

if [ "$total_files" -gt 0 ]; then
    echo "✅ 모델 파일 이동 성공!"
else
    echo "❌ 모델 파일을 찾을 수 없습니다"
fi

echo ""
echo "📋 다음 단계:"
echo "   1. python3 enhanced_test_detection.py (탐지 테스트)"
echo "   2. python3 backend/app/main.py (서버 시작)"
echo ""
echo "🎉 실제 파일 이동 완료!"