# paths.py (or top of your main/watchdog.py)
import os
from pathlib import Path

def project_root() -> Path:
    # repo root: .../WS_Monitor
    return Path(__file__).resolve().parents[0]

def default_db_path() -> Path:
    return project_root() / "resources" / "house_environment.db"

def get_db_path() -> Path:
    # systemd (or you) can override this
    p = Path(os.getenv("WSM_DB_PATH", default_db_path()))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
    
    
