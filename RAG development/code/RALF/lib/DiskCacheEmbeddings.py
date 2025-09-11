import hashlib
import pathlib
import pickle
import sqlite3
from typing import List
from langchain.embeddings.base import Embeddings

class DiskCacheEmbeddings(Embeddings):
    """
    A wrapper for an Embeddings class that caches results in a local SQLite database.
    """
    def __init__(self, inner: Embeddings, db_path: pathlib.Path):
        self.inner = inner
        self.db_path = str(db_path)
        self._ensure_db()

    def _ensure_db(self):
        """Ensures the cache table exists in the database."""
        with sqlite3.connect(self.db_path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY,vec BLOB);")

    def _key(self, text: str) -> str:
        """Create a unique key for a given text, including the inner embedder's class name."""
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{self.inner.__class__.__name__}:{h}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents, using cached results if available.
        - Fetches existing embeddings from the cache.
        - Embeds any texts that are not in the cache.
        - Stores the new embeddings in the cache.
        - Returns the full list of embeddings in the original order.
        """
        if not texts:
            return []

        keys = [self._key(t) for t in texts]
        cached_vectors = {}

        with sqlite3.connect(self.db_path) as con:
            batch_size = 500  # SQLite has a variable limit, 500 is a safe value.
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                placeholders = ",".join("?" * len(batch_keys))
                cur = con.execute(f"SELECT key, vec FROM embed_cache WHERE key IN ({placeholders})", batch_keys)
                for key, blob in cur:
                    cached_vectors[key] = pickle.loads(blob)

        missing_indices = [i for i, k in enumerate(keys) if k not in cached_vectors]
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            # This operation can be slow and may fail.
            try:
                new_vectors = self.inner.embed_documents(missing_texts)
            except Exception as e:
                print(f"Warning: Embedding of {len(missing_texts)} documents failed. Using zero-vectors. Error: {e}")
                new_vectors = []

            if new_vectors:
                to_insert = []
                for i, vec in enumerate(new_vectors):
                    key = keys[missing_indices[i]]
                    cached_vectors[key] = vec
                    to_insert.append((key, pickle.dumps(vec)))

                with sqlite3.connect(self.db_path) as con:
                    con.executemany("INSERT OR REPLACE INTO embed_cache (key, vec) VALUES (?, ?)", to_insert)

        dim = 0
        if cached_vectors:
            dim = len(next(iter(cached_vectors.values())))
        elif texts:  # If the cache was empty, try to determine dimension.
            try:
                dim = len(self.inner.embed_query(" "))
            except Exception:
                dim = 768  # Fallback if inner embedding fails.

        return [cached_vectors.get(k, [0.0] * dim) for k in keys]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query, using a cached result if available."""
        key = self._key("QUERY::" + text)
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT vec FROM embed_cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return pickle.loads(row[0])

            # Not in cache, so embed, store, and return.
            vec = self.inner.embed_query(text)
            con.execute(
                "INSERT OR REPLACE INTO embed_cache (key, vec) VALUES (?, ?)",
                (key, pickle.dumps(vec))
            )
            return vec