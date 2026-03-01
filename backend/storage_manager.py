"""
LIVEDET — Storage Manager
===========================
Handles temporary image storage with automatic cleanup.
"""

import os
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import uuid

import cv2


class StorageManager:
    """Manages temporary image storage with automatic cleanup"""

    BASE_DIR = Path(__file__).parent / "image_storage"
    ORIGINAL_DIR = BASE_DIR / "original"
    PROCESSED_DIR = BASE_DIR / "processed"

    CLEANUP_INTERVAL = 300      # 5 minutes
    FILE_RETENTION_TIME = 600   # 10 minutes

    _stored_files: Dict[str, Dict] = {}
    _cleanup_thread: Optional[threading.Thread] = None
    _cleanup_running = False

    @classmethod
    def initialize(cls):
        """Initialize storage directories and start cleanup thread."""
        cls.ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        if not cls._cleanup_running:
            cls._cleanup_running = True
            cls._cleanup_thread = threading.Thread(
                target=cls._cleanup_worker, daemon=True
            )
            cls._cleanup_thread.start()
            print("✓ Storage manager initialized with auto-cleanup")

    @classmethod
    def save_original(
        cls, image: "np.ndarray", session_id: Optional[str] = None, fmt: str = "jpg"
    ) -> Tuple[str, str]:
        """Save original image array to storage. Returns (path, session_id)."""
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        filename = f"{session_id}_original.{fmt}"
        dest = cls.ORIGINAL_DIR / filename
        cv2.imwrite(str(dest), image)

        cls._stored_files[session_id] = {
            "original": str(dest),
            "processed": None,
            "created": datetime.now(),
            "accessed": datetime.now(),
        }
        return str(dest), session_id

    @classmethod
    def save_processed(
        cls, image: "np.ndarray", session_id: str, fmt: str = "jpg"
    ) -> str:
        """Save processed/annotated image. Returns path."""
        filename = f"{session_id}_processed.{fmt}"
        dest = cls.PROCESSED_DIR / filename
        cv2.imwrite(str(dest), image)

        if session_id in cls._stored_files:
            cls._stored_files[session_id]["processed"] = str(dest)
            cls._stored_files[session_id]["accessed"] = datetime.now()
        return str(dest)

    @classmethod
    def clear_session(cls, session_id: str, delete_files: bool = True) -> bool:
        """Remove a session and optionally delete its files."""
        if session_id not in cls._stored_files:
            return False

        if delete_files:
            info = cls._stored_files[session_id]
            for key in ("original", "processed"):
                fpath = info.get(key)
                if fpath:
                    try:
                        Path(fpath).unlink(missing_ok=True)
                    except Exception:
                        pass

        del cls._stored_files[session_id]
        return True

    @classmethod
    def _cleanup_worker(cls):
        while cls._cleanup_running:
            try:
                time.sleep(cls.CLEANUP_INTERVAL)
                cls._cleanup_old_files()
            except Exception:
                pass

    @classmethod
    def _cleanup_old_files(cls):
        cutoff = datetime.now() - timedelta(seconds=cls.FILE_RETENTION_TIME)
        to_clear = [
            sid for sid, info in cls._stored_files.items()
            if info["accessed"] < cutoff
        ]
        for sid in to_clear:
            cls.clear_session(sid, delete_files=True)

    @classmethod
    def get_stats(cls) -> Dict:
        orig_size = sum(
            f.stat().st_size for f in cls.ORIGINAL_DIR.glob("*") if f.is_file()
        )
        proc_size = sum(
            f.stat().st_size for f in cls.PROCESSED_DIR.glob("*") if f.is_file()
        )
        return {
            "total_sessions": len(cls._stored_files),
            "original_files": len(list(cls.ORIGINAL_DIR.glob("*"))),
            "processed_files": len(list(cls.PROCESSED_DIR.glob("*"))),
            "original_size_mb": round(orig_size / (1024 * 1024), 2),
            "processed_size_mb": round(proc_size / (1024 * 1024), 2),
        }
