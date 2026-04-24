"""Local database cache for PMID abstracts."""
import sqlite3
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

@dataclass
class CachedAbstract:
    """Container for cached PMID abstract data."""
    pmid: str
    title: str
    abstract: str
    fetch_date: str  # ISO format timestamp


class PMIDCache:
    """SQLite-based cache for PMID abstracts."""
    
    # Thread-local storage for database connections
    _local = threading.local()
    
    def __init__(self, db_path: str = "data/pmid_cache.db"):
        """Initialize the PMID cache.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Ensure the directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database schema
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection.
        
        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create the pmid_abstracts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pmid_abstracts (
                pmid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                fetch_date TEXT NOT NULL,
                error TEXT
            )
        """)
        
        # Create an index on fetch_date for easier cleanup/maintenance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fetch_date 
            ON pmid_abstracts(fetch_date)
        """)
        
        conn.commit()
        logger.info(f"PMID cache database initialized at {self.db_path}")
    
    def get(self, pmid: str) -> Optional[CachedAbstract]:
        """Retrieve a cached abstract for a PMID.
        
        Args:
            pmid: PubMed identifier
            
        Returns:
            CachedAbstract if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pmid, title, abstract, fetch_date, error
            FROM pmid_abstracts
            WHERE pmid = ?
        """, (pmid,))
        
        row = cursor.fetchone()
        if row:
            # Only return if there's a valid abstract (no error)
            if not row['error'] and row['abstract']:
                return CachedAbstract(
                    pmid=row['pmid'],
                    title=row['title'],
                    abstract=row['abstract'],
                    fetch_date=row['fetch_date']
                )
        
        return None
    
    _SQL_VAR_LIMIT = 500

    def get_many(self, pmids: List[str]) -> Dict[str, CachedAbstract]:
        """Retrieve multiple cached abstracts.
        
        Args:
            pmids: List of PubMed identifiers
            
        Returns:
            Dictionary mapping PMID to CachedAbstract for found entries
        """
        if not pmids:
            return {}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        results = {}

        for i in range(0, len(pmids), self._SQL_VAR_LIMIT):
            chunk = pmids[i:i + self._SQL_VAR_LIMIT]
            placeholders = ','.join('?' * len(chunk))

            cursor.execute(f"""
                SELECT pmid, title, abstract, fetch_date, error
                FROM pmid_abstracts
                WHERE pmid IN ({placeholders})
                AND error IS NULL
                AND abstract IS NOT NULL
                AND abstract != ''
            """, chunk)

            for row in cursor.fetchall():
                results[row['pmid']] = CachedAbstract(
                    pmid=row['pmid'],
                    title=row['title'],
                    abstract=row['abstract'],
                    fetch_date=row['fetch_date']
                )
        
        return results
    
    def put(self, pmid: str, title: str, abstract: str, fetch_date: str, error: Optional[str] = None):
        """Store an abstract in the cache.
        
        Args:
            pmid: PubMed identifier
            title: Article title
            abstract: Article abstract
            fetch_date: ISO format timestamp
            error: Optional error message if fetch failed
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO pmid_abstracts (pmid, title, abstract, fetch_date, error)
            VALUES (?, ?, ?, ?, ?)
        """, (pmid, title, abstract, fetch_date, error))
        
        conn.commit()
    
    def put_many(self, abstracts: List[Dict]):
        """Store multiple abstracts in the cache efficiently.
        
        Args:
            abstracts: List of dictionaries with keys: pmid, title, abstract, fetch_date, error (optional)
        """
        if not abstracts:
            return
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Prepare data for executemany
        data = [
            (
                item['pmid'],
                item.get('title', ''),
                item.get('abstract', ''),
                item['fetch_date'],
                item.get('error')
            )
            for item in abstracts
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO pmid_abstracts (pmid, title, abstract, fetch_date, error)
            VALUES (?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        logger.info(f"Cached {len(abstracts)} abstracts")
    
    def has(self, pmid: str) -> bool:
        """Check if a PMID is in the cache with a valid abstract.
        
        Args:
            pmid: PubMed identifier
            
        Returns:
            True if cached with valid abstract, False otherwise
        """
        return self.get(pmid) is not None
    
    def count(self) -> int:
        """Get the total number of cached abstracts.
        
        Returns:
            Count of cached PMIDs
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM pmid_abstracts WHERE error IS NULL")
        return cursor.fetchone()[0]
    
    def get_all_cached_pmids(self) -> set:
        """Return the set of all PMIDs stored with a valid abstract."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pmid FROM pmid_abstracts
            WHERE error IS NULL AND abstract IS NOT NULL AND abstract != ''
        """)
        return {row[0] for row in cursor.fetchall()}

    def get_missing_pmids(self, pmids: List[str]) -> List[str]:
        """Get list of PMIDs that are not in the cache.
        
        For large inputs this loads all cached PMIDs once rather than
        issuing thousands of chunked IN-clause queries.
        
        Args:
            pmids: List of PubMed identifiers to check
            
        Returns:
            List of PMIDs not found in cache
        """
        if len(pmids) > self._SQL_VAR_LIMIT:
            cached_set = self.get_all_cached_pmids()
            return [p for p in pmids if p not in cached_set]

        cached = self.get_many(pmids)
        return [pmid for pmid in pmids if pmid not in cached]
    
    def clear(self):
        """Clear all cached abstracts (use with caution)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM pmid_abstracts")
        conn.commit()
        logger.warning("Cleared all cached abstracts")
    
    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

