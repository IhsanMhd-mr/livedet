"""
LIVEDET Configuration
Reads from .env file for easy cross-device configuration
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Load configuration from .env file"""

    def __init__(self, env_file: Optional[str] = None):
        if env_file is None:
            env_file = Path(__file__).parent.parent / ".env"

        self.env_file = Path(env_file)
        self._load_env_file()
        self._load_environment_variables()

    def _load_env_file(self):
        """Load .env file into os.environ"""
        if not self.env_file.exists():
            logger.warning(f".env file not found at {self.env_file}")
            logger.info("Using default values. Create .env from .env.example")
            return

        try:
            with open(self.env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key not in os.environ:
                            os.environ[key] = value

            logger.info(f"Loaded configuration from {self.env_file}")
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")

    def _load_environment_variables(self):
        """Load all configuration variables"""
        # Flask settings
        self.FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
        self.FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))
        self.FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        # Model configuration
        self.MODEL_TYPE = os.getenv("MODEL_TYPE", "yolov8s")
        self.BEST_MODEL_PATH = os.getenv(
            "BEST_MODEL_PATH",
            str(Path(__file__).parent.parent / "models" / "livedet_best.pt"),
        )
        self.ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "BEST_MODEL")

        # Inference settings
        self.DEVICE = os.getenv("DEVICE", "cpu")
        self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

        # Workers
        self.NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

    def get(self, key: str, default=None):
        """Get configuration value by key"""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return "\n".join([f"{k}: {v}" for k, v in sorted(attrs.items())])


# Global config instance
config = Config()

if __name__ == "__main__":
    print("LIVEDET Configuration:")
    print("=" * 60)
    print(config)
