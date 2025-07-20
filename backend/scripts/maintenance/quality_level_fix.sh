#!/bin/bash

echo "🔧 QualityLevel.MAXIMUM → QualityLevel.ULTRA 통일화"
echo "=============================================="

cd backend

# 수정할 파일들 찾기
echo "🔍 수정 대상 파일 검색 중..."

files_to_fix=(
    "app/ai_pipeline/pipeline_manager.py"
    "app/ai_pipeline/steps/step_07_post_processing.py"
    "app/core/config.py"
    "app/core/pipeline_config.py"
    "app/config.py"
)

echo "📝 발견된 파일들:"
for file in "${files_to_fix[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (없음)"
    fi
done

echo ""
echo "🔧 QualityLevel Enum 수정 시작..."

# 각 파일별로 수정
for file in "${files_to_fix[@]}"; do
    if [ -f "$file" ]; then
        echo "  📝 $file 수정 중..."
        
        # 백업 생성
        cp "$file" "$file.backup_$(date +%Y%m%d_%H%M%S)"
        
        # 1. MAXIMUM을 ULTRA로 변경
        sed -i '' 's/QualityLevel\.MAXIMUM/QualityLevel.ULTRA/g' "$file"
        sed -i '' 's/MAXIMUM = "maximum"/ULTRA = "ultra"/g' "$file"
        
        # 2. 문자열 값들도 변경
        sed -i '' 's/"maximum"/"ultra"/g' "$file"
        sed -i '' "s/'maximum'/'ultra'/g" "$file"
        
        # 3. 주석이나 설명에서도 변경
        sed -i '' 's/maximum"/ultra"/g' "$file"
        sed -i '' "s/maximum'/ultra'/g" "$file"
        
        # 4. performance_mode 같은 다른 설정은 그대로 유지하고 quality_level만 변경
        sed -i '' 's/quality_level.*=.*"maximum"/quality_level = "ultra"/g' "$file"
        
        echo "    ✅ $file 수정 완료"
    fi
done

echo ""
echo "🧪 수정 결과 확인..."

# 수정된 내용 확인
for file in "${files_to_fix[@]}"; do
    if [ -f "$file" ]; then
        echo "📋 $file:"
        
        # MAXIMUM이 남아있는지 확인
        maximum_count=$(grep -c "MAXIMUM\|maximum" "$file" 2>/dev/null || echo "0")
        ultra_count=$(grep -c "ULTRA\|ultra" "$file" 2>/dev/null || echo "0")
        
        echo "  - MAXIMUM/maximum 남은 개수: $maximum_count"
        echo "  - ULTRA/ultra 개수: $ultra_count"
        
        if [ "$maximum_count" -gt 0 ]; then
            echo "  ⚠️ 아직 MAXIMUM이 남아있습니다:"
            grep -n "MAXIMUM\|maximum" "$file" | head -3
        else
            echo "  ✅ MAXIMUM → ULTRA 변경 완료"
        fi
    fi
done

echo ""
echo "🎯 QualityLevel Enum 정의 확인..."

# QualityLevel 클래스 정의가 있는 파일들 확인
find app/ -name "*.py" -exec grep -l "class QualityLevel" {} \; 2>/dev/null | while read file; do
    echo "📋 $file의 QualityLevel 정의:"
    grep -A 10 "class QualityLevel" "$file" | grep -E "(FAST|BALANCED|HIGH|ULTRA|MAXIMUM)" || echo "  정의를 찾을 수 없음"
done

echo ""
echo "✅ QualityLevel.MAXIMUM → QualityLevel.ULTRA 수정 완료!"
echo "========================================="
echo ""
echo "🚀 이제 서버를 실행해보세요:"
echo "   python app/main.py"