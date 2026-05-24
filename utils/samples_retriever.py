"""
Sample Retriever Module

Provides embedding storage, retrieval, and similarity calculation.
Uses SQLite with LRU cache for efficient embedding management.
"""
import difflib
import json
import os
import sqlite3
import sys
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from call_llm_api import call_embedding_general

CACHE_SIZE = 10000
query_embeddings_path = "query_embeddings.jsonl"
query_index_path = "query_embeddings.idx"


class LRUCache:
    """Thread-safe LRU cache"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[List[float]]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: List[float]):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache


class EmbeddingStore:
    """
    Embedding storage: SQLite direct storage + LRU in-memory cache.
    """
    def __init__(self, jsonl_path: str, index_path: str, cache_size: int = CACHE_SIZE):
        self.jsonl_path = jsonl_path
        self.index_path = index_path
        self.cache = LRUCache(cache_size)
        self.write_lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)

        self.db = sqlite3.connect(index_path, check_same_thread=False, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db_lock = threading.Lock()
        self._init_db()
        self._migrate_from_jsonl_if_needed()
        self._count = self._get_count()
        print(f"[EmbeddingStore] init success， {self._count} records")

    def _init_db(self):
        """Initialize database table schema."""
        cursor = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_index'")
        if cursor.fetchone() is not None:
            cursor = self.db.execute("PRAGMA table_info(query_index)")
            columns = {row[1] for row in cursor.fetchall()}
            if 'embedding' not in columns:
                print("[EmbeddingStore] find old table")
                self.db.execute('DROP TABLE query_index')
                self.db.commit()

        self.db.execute('''
            CREATE TABLE IF NOT EXISTS query_index (
                query TEXT PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        ''')
        self.db.commit()

    def _migrate_from_jsonl_if_needed(self):
        """Migrate data from legacy JSONL file."""
        if not os.path.exists(self.jsonl_path):
            return
        jsonl_size = os.path.getsize(self.jsonl_path)
        if jsonl_size == 0:
            return

        if self._get_count() > 0:
            return

        with self.db_lock:
            if self._get_count_unlocked() > 0:
                return
            print(f"[EmbeddingStore] loading from JSONL")
            count = 0
            try:
                with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            query_key = data.get('query', '').lower()
                            embedding = data.get('embedding')
                            if query_key and embedding:
                                blob = np.array(embedding, dtype=np.float32).tobytes()
                                self.db.execute(
                                    'INSERT OR IGNORE INTO query_index (query, embedding) VALUES (?, ?)',
                                    (query_key, blob)
                                )
                                count += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                self.db.commit()
                print(f"[EmbeddingStore] ， {count} records")
                backup_path = self.jsonl_path + ".migrated"
                try:
                    os.rename(self.jsonl_path, backup_path)
                except OSError:
                    pass
            except IOError as e:
                print(f"[Error] : {e}")

    def _get_count(self) -> int:
        with self.db_lock:
            return self._get_count_unlocked()

    def _get_count_unlocked(self) -> int:
        cursor = self.db.execute('SELECT COUNT(*) FROM query_index')
        return cursor.fetchone()[0]

    def get(self, query: str) -> Optional[List[float]]:
        """Get embedding: check LRU cache first, then SQLite on miss."""
        query_key = query.lower()

        cached = self.cache.get(query_key)
        if cached is not None:
            return cached

        with self.db_lock:
            cursor = self.db.execute(
                'SELECT embedding FROM query_index WHERE query = ?',
                (query_key,)
            )
            row = cursor.fetchone()
        if row is None:
            return None

        try:
            embedding = np.frombuffer(row[0], dtype=np.float32).tolist()
            self.cache.put(query_key, embedding)
            return embedding
        except Exception as e:
            print(f"[Warning]: {e}")
            return None

    def put(self, query: str, embedding: List[float]):
        """Write embedding to SQLite."""
        query_key = query.lower()

        if self.get(query) is not None:
            return

        with self.write_lock:
            if self.get(query) is not None:
                return

            blob = np.array(embedding, dtype=np.float32).tobytes()

            with self.db_lock:
                self.db.execute(
                    'INSERT OR IGNORE INTO query_index (query, embedding) VALUES (?, ?)',
                    (query_key, blob)
                )
                self.db.commit()

            self.cache.put(query_key, embedding)
            self._count += 1

    def close(self):
        self.db.close()


_store = EmbeddingStore(query_embeddings_path, query_index_path)

_embedding_in_flight = {}
_embedding_flight_lock = threading.Lock()


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1, vec2: input vectors

    Returns:
        float: similarity score [-1, 1]
    """
    try:
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)
    except ValueError as e:
        print(f"[Error] 向量转换失败: {e}")
        return 0

    if vec1.shape != vec2.shape:
        print(f"[Error] 向量维度不一致: {vec1.shape} vs {vec2.shape}")
        return 0

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return float(np.clip(similarity, -1.0, 1.0))


def get_or_create_embedding(query: str) -> List[float]:
    """
    Get or create embedding.
    Uses single-flight pattern to prevent concurrent requests for the same query embedding.

    Args:
        query: query text

    Returns:
        List[float]: embedding vector, returns 0 on failure
    """
    global _embedding_in_flight, _embedding_flight_lock

    query_key = query.lower()

    embedding = _store.get(query_key)
    if embedding is not None:
        return embedding

    with _embedding_flight_lock:
        if query_key in _embedding_in_flight:
            event = _embedding_in_flight[query_key]
        else:
            event = threading.Event()
            _embedding_in_flight[query_key] = event
            event = None

    if event is not None:
        finished = event.wait(timeout=120)
        if not finished:
            print(f"[Warning] wait embedding timeout")
            with _embedding_flight_lock:
                _embedding_in_flight.pop(query_key, None)
            return 0
        embedding = _store.get(query_key)
        return embedding if embedding is not None else 0

    try:
        embedding_result = call_embedding_general(
            query_key,
            model="text-embedding-3-small",
            app_name="ant_or_annotation"
        )

        if "embeddings" not in embedding_result or not embedding_result["embeddings"]:
            return 0

        embedding = embedding_result["embeddings"][0]
        _store.put(query_key, embedding)
        return embedding

    except Exception as e:
        print(f"[Error] find embedding failed: {e}")
        return 0
    finally:
        with _embedding_flight_lock:
            event = _embedding_in_flight.pop(query_key, None)
            if event is not None:
                event.set()
