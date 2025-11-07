"""Database layer for ATS Brain with parameterized queries and safe connection management."""
from typing import Optional, Dict, Any
import pyodbc
from contextlib import contextmanager
from loguru import logger
import json
import base64

from .config import settings


@contextmanager
def get_connection():
    """Context manager for database connections with automatic cleanup."""
    if not settings.db_conn:
        raise RuntimeError("DB_CONN environment variable not configured")
    conn = pyodbc.connect(settings.db_conn)
    try:
        yield conn
    except Exception as e:
        logger.error("Database error: {}", e)
        conn.rollback()
        raise
    finally:
        conn.close()


def get_job_posting(job_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch job posting from dbo.JobPostings by job_id.
    Returns dict with JobID, JobTitle, JDText, JDFile, CreatedAt.
    """
    query = (
        "SELECT TOP 1 JobID, JobTitle, JDText, JDFile, CreatedAt "
        "FROM dbo.JobPostings WHERE JobID = ?"
    )
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "JobID": row.JobID,
                "JobTitle": getattr(row, "JobTitle", None),
                "JDText": getattr(row, "JDText", None),
                "JDFile": getattr(row, "JDFile", None),
                "CreatedAt": getattr(row, "CreatedAt", None),
            }
    except Exception as e:
        logger.error("Failed to fetch job posting job_id={}: {}", job_id, e)
        raise


def get_resume_blob(resume_id: int) -> Optional[bytes]:
    """
    Fetch resume blob (PDF/DOCX) from dbo.Resume by resume_id.
    Returns the raw bytes of the resume file.
    """
    query = "SELECT TOP 1 ResumeBlob FROM dbo.Resume WHERE ResumeID = ?"
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (resume_id,))
            row = cursor.fetchone()
            if not row:
                return None
            blob = getattr(row, "ResumeBlob", None)
            if blob is None:
                return None
            # pyodbc returns bytes for VARBINARY(MAX) columns
            if isinstance(blob, bytes):
                return blob
            # Fallback if it's a string or other type
            return bytes(blob) if blob else None
    except Exception as e:
        logger.error("Failed to fetch resume blob resume_id={}: {}", resume_id, e)
        raise


def get_resume_metadata(resume_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch resume metadata from dbo.Resume by resume_id.
    Returns dict with CandidateName and other metadata fields if available.
    """
    query = (
        "SELECT TOP 1 ResumeID, CandidateName, CreatedAt "
        "FROM dbo.Resume WHERE ResumeID = ?"
    )
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (resume_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "ResumeID": row.ResumeID,
                "CandidateName": getattr(row, "CandidateName", None),
                "CreatedAt": getattr(row, "CreatedAt", None),
            }
    except Exception as e:
        logger.warning("Failed to fetch resume metadata resume_id={}: {}", resume_id, e)
        # Return None if metadata not available (don't fail the whole operation)
        return None


def get_parsed_jd(job_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch parsed JD from dbo.ParsedData (Type='JD', JobID=job_id).
    Returns dict with ID, Type, JobID, ResumeID, ParsedJSON, MatchScore, Embedding, CreatedAt.
    """
    query = (
        "SELECT TOP 1 ID, Type, JobID, ResumeID, ParsedJSON, MatchScore, Embedding, CreatedAt "
        "FROM dbo.ParsedData WHERE Type = 'JD' AND JobID = ? ORDER BY CreatedAt DESC"
    )
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            embedding = getattr(row, "Embedding", None)
            # Decode embedding if it's stored as base64 string in ParsedJSON
            # or if Embedding column contains bytes
            embedding_data = None
            if embedding:
                if isinstance(embedding, bytes):
                    embedding_data = base64.b64encode(embedding).decode('utf-8')
                elif isinstance(embedding, str):
                    embedding_data = embedding
            
            return {
                "ID": row.ID,
                "Type": row.Type,
                "JobID": row.JobID,
                "ResumeID": getattr(row, "ResumeID", None),
                "ParsedJSON": getattr(row, "ParsedJSON", None),
                "MatchScore": getattr(row, "MatchScore", None),
                "Embedding": embedding_data,
                "CreatedAt": getattr(row, "CreatedAt", None),
            }
    except Exception as e:
        logger.error("Failed to fetch parsed JD job_id={}: {}", job_id, e)
        raise


def insert_parsed_data(record_dict: Dict[str, Any]) -> None:
    """
    Insert parsed data record into dbo.ParsedData.
    
    Args:
        record_dict: Dictionary with keys:
            - Type: 'JD' or 'Resume'
            - JobID: int
            - ResumeID: int or None
            - ParsedJSON: JSON string (nvarchar(max))
            - MatchScore: float or None
            - Embedding: bytes or base64 string or None (stored in varbinary(max) or in ParsedJSON)
    
    Note: Embedding can be stored in two ways:
    1. In separate Embedding column (varbinary(max)) - recommended for production
    2. As base64 string in ParsedJSON - for simplicity or when column doesn't exist
    """
    # Option 1: Store embedding in separate column (recommended)
    # Option 2: Store as base64 in ParsedJSON (fallback)
    embedding = record_dict.get("Embedding")
    embedding_bytes = None
    
    if embedding:
        if isinstance(embedding, bytes):
            embedding_bytes = embedding
        elif isinstance(embedding, str):
            # Assume base64 string, decode to bytes
            try:
                embedding_bytes = base64.b64decode(embedding)
            except Exception:
                logger.warning("Failed to decode embedding from base64, storing as None")
                embedding_bytes = None
    
    # If embedding_bytes exists, store in Embedding column; otherwise store base64 in ParsedJSON
    if embedding_bytes:
        query = (
            "INSERT INTO dbo.ParsedData (Type, JobID, ResumeID, ParsedJSON, MatchScore, Embedding, CreatedAt) "
            "VALUES (?, ?, ?, ?, ?, ?, GETDATE())"
        )
        params = (
            record_dict.get("Type"),
            record_dict.get("JobID"),
            record_dict.get("ResumeID"),
            record_dict.get("ParsedJSON"),
            record_dict.get("MatchScore"),
            embedding_bytes,
        )
    else:
        # Fallback: store embedding as base64 in ParsedJSON if provided
        parsed_json_dict = {}
        if record_dict.get("ParsedJSON"):
            try:
                parsed_json_dict = json.loads(record_dict["ParsedJSON"])
            except Exception:
                parsed_json_dict = {}
        
        if embedding and isinstance(embedding, str):
            parsed_json_dict["embedding_base64"] = embedding
        
        query = (
            "INSERT INTO dbo.ParsedData (Type, JobID, ResumeID, ParsedJSON, MatchScore, Embedding, CreatedAt) "
            "VALUES (?, ?, ?, ?, ?, ?, GETDATE())"
        )
        params = (
            record_dict.get("Type"),
            record_dict.get("JobID"),
            record_dict.get("ResumeID"),
            json.dumps(parsed_json_dict, ensure_ascii=False),
            record_dict.get("MatchScore"),
            None,  # Embedding column is NULL
        )
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            logger.info(
                "Inserted parsed data: type={}, job_id={}, resume_id={}",
                record_dict.get("Type"),
                record_dict.get("JobID"),
                record_dict.get("ResumeID"),
            )
    except Exception as e:
        logger.error("Failed to insert parsed data: {}", e)
        raise


def get_match_result(match_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch match result from dbo.ParsedData by match_id (stored in ParsedJSON).
    Returns dict with ID, Type, JobID, ResumeID, ParsedJSON, MatchScore, Embedding, CreatedAt.
    """
    query = (
        "SELECT TOP 1 ID, Type, JobID, ResumeID, ParsedJSON, MatchScore, Embedding, CreatedAt "
        "FROM dbo.ParsedData WHERE ParsedJSON LIKE ? ORDER BY CreatedAt DESC"
    )
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            # Search for match_id in ParsedJSON
            cursor.execute(query, (f"%{match_id}%",))
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse ParsedJSON to verify match_id
            try:
                parsed_json = json.loads(row.ParsedJSON)
                if parsed_json.get("match_id") != match_id:
                    return None
            except Exception:
                return None
            
            return {
                "ID": row.ID,
                "Type": row.Type,
                "JobID": row.JobID,
                "ResumeID": getattr(row, "ResumeID", None),
                "ParsedJSON": row.ParsedJSON,
                "MatchScore": getattr(row, "MatchScore", None),
                "Embedding": getattr(row, "Embedding", None),
                "CreatedAt": getattr(row, "CreatedAt", None),
            }
    except Exception as e:
        logger.error("Failed to fetch match result match_id={}: {}", match_id, e)
        raise
