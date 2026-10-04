"""Settings: reads .env (if present) into the environment, plus paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("VSP_DATA", ROOT / "data"))
WEB = ROOT / "web"


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments. Never overrides real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
DATA.mkdir(parents=True, exist_ok=True)
