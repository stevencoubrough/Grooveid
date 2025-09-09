import sqlite3, hashlib, time, struct, os
import numpy as np
from contextlib import contextmanager

DB_PATH = os.getenv("EMBED_CACHE_PATH", "/tmp/embed_cache.sqlite3")
TTL_SECONDS = int(os.getenv("EMBED_CACHE_TTL", "1209600"))  # 14 days

def _ensure_schema(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS url_vectors(
        url_hash TEXT PRIMARY KEY,
        dim INTEGER NOT NULL,
        vec BLOB NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON url_vectors(updated_at)")
    conn.commit()

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()

def _h(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()

def save(url: str, vec: np.ndarray):
    assert vec.dtype in (np.float16, np.float32)
    v = vec.astype(np.float16, copy=False)  # compress to fp16
    blob = v.tobytes()
    now = int(time.time())
    with _db() as conn:
        conn.execute(
            "REPLACE INTO url_vectors(url_hash, dim, vec, updated_at) VALUES(?,?,?,?)",
            (_h(url), v.shape[0], sqlite3.Binary(blob), now),
        )
        conn.commit()

def load(url: str) -> np.ndarray | None:
    key = _h(url)
    cutoff = int(time.time()) - TTL_SECONDS
    with _db() as conn:
        row = conn.execute(
            "SELECT dim, vec, updated_at FROM url_vectors WHERE url_hash=?",
            (key,),
        ).fetchone()
        if not row:
            return None
        dim, blob, updated = row
        if updated < cutoff:
            try:
                conn.execute("DELETE FROM url_vectors WHERE url_hash=?", (key,))
            except:
                pass
            return None
        arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        return arr.reshape((dim,))

def vacuum_old():
    cutoff = int(time.time()) - TTL_SECONDS
    with _db() as conn:
        conn.execute("DELETE FROM url_vectors WHERE updated_at < ?", (cutoff,))
        conn.commit()
