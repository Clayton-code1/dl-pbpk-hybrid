from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/dlpbpk"
)
API_KEY: str = os.getenv("API_KEY", "")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
