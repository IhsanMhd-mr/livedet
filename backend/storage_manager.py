"""
Storage Manager - Handle temporary image storage and cleanup
Manages incoming images, processed results, and automatic cleanup
"""

import os
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import uuid


class StorageManager:
    """Manages temporary image storage with automatic cleanup"""
    
    # Storage paths
    BASE_DIR = Path(__file__).parent / "image_storage"
    ORIGINAL_DIR = BASE_DIR / "original"
    PROCESSED_DIR = BASE_DIR / "processed"
    
    # Cleanup settings
    CLEANUP_INTERVAL = 300  # 5 minutes
    FILE_RETENTION_TIME = 600  # 10 minutes (files older than this are deleted)
    
    # Track stored files
    _stored_files: Dict[str, Dict] = {}
    _cleanup_thread: Optional[threading.Thread] = None
    _cleanup_running = False
    
    @classmethod
    def initialize(cls):
        """Initialize storage directories and start cleanup thread"""
        cls.ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup thread if not already running
        if not cls._cleanup_running:
            cls._cleanup_running = True
            cls._cleanup_thread = threading.Thread(
                target=cls._cleanup_worker,
                daemon=True
            )
            cls._cleanup_thread.start()
            print("✓ Storage manager initialized with auto-cleanup enabled")
    
    @classmethod
    def save_original_image(cls, image_path: str, session_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Save uploaded image to original folder
        
        Args:
            image_path: Path to uploaded image
            session_id: Optional session ID for tracking
            
        Returns:
            Tuple of (new_path, session_id)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        # Create unique filename with session ID
        original_filename = Path(image_path).stem
        ext = Path(image_path).suffix
        unique_filename = f"{session_id}_{original_filename}{ext}"
        
        # Save to original folder
        dest_path = cls.ORIGINAL_DIR / unique_filename
        shutil.copy2(image_path, dest_path)
        
        # Track file
        cls._stored_files[session_id] = {
            'original': str(dest_path),
            'processed': None,
            'created': datetime.now(),
            'accessed': datetime.now()
        }
        
        print(f"✓ Original image saved: {dest_path}")
        return str(dest_path), session_id
    
    @classmethod
    def save_original_image_array(
        cls,
        image_data,
        session_id: Optional[str] = None,
        image_format: str = "jpg"
    ) -> Tuple[str, str]:
        """
        Save numpy array as original image to storage
        
        Args:
            image_data: Image array (numpy/cv2 format)
            session_id: Optional session ID for tracking
            image_format: Image format (jpg, png, etc)
            
        Returns:
            Tuple of (path, session_id)
        """
        import cv2
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        # Create filename
        original_filename = f"{session_id}_original.{image_format}"
        dest_path = cls.ORIGINAL_DIR / original_filename
        
        # Save image
        cv2.imwrite(str(dest_path), image_data)
        
        # Track file
        cls._stored_files[session_id] = {
            'original': str(dest_path),
            'processed': None,
            'created': datetime.now(),
            'accessed': datetime.now()
        }
        
        print(f"✓ Original image saved: {dest_path}")
        return str(dest_path), session_id
    
    @classmethod
    def save_processed_image(
        cls,
        image_data,
        session_id: str,
        image_format: str = "jpg"
    ) -> str:
        """
        Save processed image to processed folder
        
        Args:
            image_data: Image array (numpy/cv2 format)
            session_id: Session ID for linking to original
            image_format: Image format (jpg, png, etc)
            
        Returns:
            Path to saved processed image
        """
        import cv2
        
        if session_id not in cls._stored_files:
            raise ValueError(f"Session ID not found: {session_id}")
        
        # Create filename
        processed_filename = f"{session_id}_processed.{image_format}"
        dest_path = cls.PROCESSED_DIR / processed_filename
        
        # Save image
        cv2.imwrite(str(dest_path), image_data)
        
        # Update tracking
        cls._stored_files[session_id]['processed'] = str(dest_path)
        cls._stored_files[session_id]['accessed'] = datetime.now()
        
        print(f"✓ Processed image saved: {dest_path}")
        return str(dest_path)
    
    @classmethod
    def save_json_report(
        cls,
        report_data: dict,
        session_id: str,
        report_name: str = "report"
    ) -> str:
        """
        Save JSON report to processed folder
        
        Args:
            report_data: Dictionary with analysis data
            session_id: Session ID for linking
            report_name: Name for the report file
            
        Returns:
            Path to saved JSON report
        """
        import json
        
        if session_id not in cls._stored_files:
            raise ValueError(f"Session ID not found: {session_id}")
        
        # Create filename
        report_filename = f"{session_id}_{report_name}.json"
        dest_path = cls.PROCESSED_DIR / report_filename
        
        # Save JSON
        with open(dest_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        cls._stored_files[session_id]['accessed'] = datetime.now()
        
        print(f"✓ Report saved: {dest_path}")
        return str(dest_path)
    
    @classmethod
    def get_session_info(cls, session_id: str) -> Optional[Dict]:
        """Get information about a stored session"""
        return cls._stored_files.get(session_id)
    
    @classmethod
    def get_original_image_path(cls, session_id: str) -> Optional[str]:
        """Get path to original image for a session"""
        if session_id in cls._stored_files:
            return cls._stored_files[session_id]['original']
        return None
    
    @classmethod
    def get_processed_image_path(cls, session_id: str) -> Optional[str]:
        """Get path to processed image for a session"""
        if session_id in cls._stored_files:
            return cls._stored_files[session_id]['processed']
        return None
    
    @classmethod
    def clear_session(cls, session_id: str, delete_files: bool = True) -> bool:
        """
        Clear session and optionally delete files
        
        Args:
            session_id: Session ID to clear
            delete_files: Whether to physically delete files
            
        Returns:
            True if successful
        """
        if session_id not in cls._stored_files:
            return False
        
        if delete_files:
            session_info = cls._stored_files[session_id]
            
            # Delete original
            if session_info['original']:
                try:
                    Path(session_info['original']).unlink()
                    print(f"✓ Deleted original: {session_info['original']}")
                except Exception as e:
                    print(f"✗ Error deleting original: {e}")
            
            # Delete processed
            if session_info['processed']:
                try:
                    Path(session_info['processed']).unlink()
                    print(f"✓ Deleted processed: {session_info['processed']}")
                except Exception as e:
                    print(f"✗ Error deleting processed: {e}")
        
        # Remove from tracking
        del cls._stored_files[session_id]
        return True
    
    @classmethod
    def _cleanup_worker(cls):
        """Background cleanup worker - runs periodically"""
        while cls._cleanup_running:
            try:
                time.sleep(cls.CLEANUP_INTERVAL)
                cls._cleanup_old_files()
            except Exception as e:
                print(f"✗ Cleanup worker error: {e}")
    
    @classmethod
    def _cleanup_old_files(cls):
        """Delete files older than retention time"""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=cls.FILE_RETENTION_TIME)
        
        sessions_to_clear = []
        
        for session_id, info in cls._stored_files.items():
            accessed_time = info['accessed']
            
            if accessed_time < cutoff_time:
                sessions_to_clear.append(session_id)
        
        for session_id in sessions_to_clear:
            print(f" Auto-cleanup: Clearing session {session_id}")
            cls.clear_session(session_id, delete_files=True)
    
    @classmethod
    def cleanup_all(cls):
        """Clear all sessions and stop cleanup thread"""
        cls._cleanup_running = False
        
        # Clear all sessions
        session_ids = list(cls._stored_files.keys())
        for session_id in session_ids:
            cls.clear_session(session_id, delete_files=True)
        
        print("✓ Storage cleanup complete")
    
    @classmethod
    def get_storage_stats(cls) -> Dict:
        """Get storage statistics"""
        original_size = sum(
            f.stat().st_size for f in cls.ORIGINAL_DIR.glob("*")
            if f.is_file()
        )
        processed_size = sum(
            f.stat().st_size for f in cls.PROCESSED_DIR.glob("*")
            if f.is_file()
        )
        
        return {
            'total_sessions': len(cls._stored_files),
            'original_files': len(list(cls.ORIGINAL_DIR.glob("*"))),
            'processed_files': len(list(cls.PROCESSED_DIR.glob("*"))),
            'original_size_mb': round(original_size / (1024 * 1024), 2),
            'processed_size_mb': round(processed_size / (1024 * 1024), 2),
            'cleanup_running': cls._cleanup_running
        }


# Initialize on import
StorageManager.initialize()
