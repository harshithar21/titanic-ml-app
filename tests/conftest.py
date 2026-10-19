import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    # train a model first if the saved file is missing (e.g. on a fresh CI machine)
    if not (ROOT / "models" / "titanic_pipeline.joblib").exists():
        from ml.train import run
        run(plots=False)
    from app.main import app
    with TestClient(app) as c:   # the `with` block triggers the startup event that loads the model
        yield c
