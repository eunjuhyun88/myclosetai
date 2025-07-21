from pathlib import Path
import os

BACKEND_DIR = Path(__file__).parent.parent.parent
AI_MODELS_DIR = BACKEND_DIR / "ai_models"

os.environ.setdefault("MYCLOSET_AI_MODELS_PATH", str(AI_MODELS_DIR))

def get_ai_models_path():
    return AI_MODELS_DIR

print(f"🔧 AI Models 경로: {AI_MODELS_DIR}")
print(f"📁 경로 존재: {AI_MODELS_DIR.exists()}")
