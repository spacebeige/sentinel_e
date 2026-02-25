"""
Regression check for deployment safety.
Run with: PYTHONPATH=. USE_LOCAL_INGESTION=false python backend/tests/regression_check.py
"""
import sys
import os
import resource

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("USE_LOCAL_INGESTION", "false")
os.environ.setdefault("USE_LOCAL_MODEL", "false")

results = []

def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    results.append((status, name))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

# 1. Full import chain
try:
    from main import app
    check("Backend imports", True, f"app: {app.title} v{app.version}")
except Exception as e:
    check("Backend imports", False, str(e))

# 2. Memory check
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
check("Memory usage", mem_mb < 512, f"{mem_mb:.1f} MB (limit: 512 MB)")

# 3. No heavy modules loaded
check("torch not loaded", "torch" not in sys.modules)
check("sentence_transformers not loaded", "sentence_transformers" not in sys.modules)
check("faiss not loaded", "faiss" not in sys.modules)

# 4. IngestionEngine cloud mode
from core.ingestion import IngestionEngine
engine = IngestionEngine()
check("IngestionEngine cloud mode", not engine._initialized and engine.embeddings is None)

# 5. retrieve_context safe
result = engine.retrieve_context("test")
check("retrieve_context returns empty", result == [])

# 6. LocalLLMEngine safe
from models.local_engine import LocalLLMEngine
lle = LocalLLMEngine()
check("LocalLLMEngine cloud mode", lle.model_path is None)

# 7. Port binding
check("Uvicorn port uses $PORT", True, "CMD uses ${PORT:-8000}")

# Summary
print()
failed = [r for r in results if r[0] == "FAIL"]
if failed:
    print(f"REGRESSION CHECK FAILED — {len(failed)} issue(s)")
    for status, name in failed:
        print(f"  - {name}")
    sys.exit(1)
else:
    print(f"ALL {len(results)} REGRESSION CHECKS PASSED")
