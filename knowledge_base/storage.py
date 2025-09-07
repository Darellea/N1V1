"""
Knowledge Base Storage Layer

This module handles the persistence of knowledge entries to various storage backends
including JSON, CSV, and SQLite. It provides read/write operations, indexing,
and versioning of stored knowledge.
"""

from __future__ import annotations

import json
import csv
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import hashlib
import gzip
import pickle
from abc import ABC, abstractmethod

from .schema import KnowledgeEntry, KnowledgeQuery, KnowledgeQueryResult, validate_knowledge_entry


class StorageBackend(ABC):
    """Abstract base class for knowledge storage backends."""

    @abstractmethod
    def save_entry(self, entry: KnowledgeEntry) -> bool:
        """Save a knowledge entry."""
        pass

    @abstractmethod
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a knowledge entry by ID."""
        pass

    @abstractmethod
    def query_entries(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """Query knowledge entries based on criteria."""
        pass

    @abstractmethod
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        pass

    @abstractmethod
    def list_entries(self, limit: int = 100) -> List[KnowledgeEntry]:
        """List all knowledge entries."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up resources."""
        pass


class JSONStorage(StorageBackend):
    """JSON file-based storage backend."""

    def __init__(self, file_path: Union[str, Path], compress: bool = False):
        self.file_path = Path(file_path)
        self.compress = compress
        self._lock = threading.RLock()
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._load_entries()

    def _load_entries(self):
        """Load entries from JSON file."""
        if not self.file_path.exists():
            return

        try:
            with self._lock:
                if self.compress:
                    with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(self.file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                for entry_data in data.get('entries', []):
                    try:
                        entry = KnowledgeEntry.from_dict(entry_data)
                        self._entries[entry.id] = entry
                    except Exception as e:
                        print(f"Failed to load entry {entry_data.get('id', 'unknown')}: {e}")

        except (json.JSONDecodeError, FileNotFoundError, gzip.BadGzipFile) as e:
            print(f"Failed to load knowledge base: {e}")
            self._entries = {}

    def _save_entries(self):
        """Save entries to JSON file."""
        try:
            with self._lock:
                data = {
                    'version': '1.0',
                    'last_updated': datetime.now().isoformat(),
                    'entries': [entry.to_dict() for entry in self._entries.values()]
                }

                self.file_path.parent.mkdir(parents=True, exist_ok=True)

                if self.compress:
                    with gzip.open(self.file_path, 'wt', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                else:
                    with open(self.file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Failed to save knowledge base: {e}")
            return False
        return True

    def save_entry(self, entry: KnowledgeEntry) -> bool:
        """Save a knowledge entry."""
        errors = validate_knowledge_entry(entry)
        if errors:
            print(f"Invalid entry: {errors}")
            return False

        with self._lock:
            self._entries[entry.id] = entry
            return self._save_entries()

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a knowledge entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def query_entries(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """Query knowledge entries based on criteria."""
        import time
        start_time = time.time()

        with self._lock:
            candidates = list(self._entries.values())

        # Apply filters
        filtered = []
        for entry in candidates:
            if query.market_regime and entry.market_condition.regime != query.market_regime:
                continue
            if query.strategy_name and entry.strategy_metadata.name != query.strategy_name:
                continue
            if query.strategy_category and entry.strategy_metadata.category != query.strategy_category:
                continue
            if entry.confidence_score < query.min_confidence:
                continue
            if entry.sample_size < query.min_sample_size:
                continue
            if query.timeframe and entry.strategy_metadata.timeframe != query.timeframe:
                continue
            if query.tags:
                if not any(tag in entry.tags for tag in query.tags):
                    continue

            filtered.append(entry)

        # Sort results
        if query.sort_by == "confidence_score":
            filtered.sort(key=lambda x: x.confidence_score, reverse=True)
        elif query.sort_by == "win_rate":
            filtered.sort(key=lambda x: x.performance.win_rate, reverse=True)
        elif query.sort_by == "profit_factor":
            filtered.sort(key=lambda x: x.performance.profit_factor, reverse=True)
        elif query.sort_by == "sample_size":
            filtered.sort(key=lambda x: x.sample_size, reverse=True)

        # Apply limit
        results = filtered[:query.limit]

        execution_time = time.time() - start_time
        return KnowledgeQueryResult(results, len(filtered), query, execution_time)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                return self._save_entries()
        return False

    def list_entries(self, limit: int = 100) -> List[KnowledgeEntry]:
        """List all knowledge entries."""
        with self._lock:
            entries = list(self._entries.values())
            return entries[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_entries = len(self._entries)
            avg_confidence = sum(e.confidence_score for e in self._entries.values()) / total_entries if total_entries > 0 else 0
            avg_sample_size = sum(e.sample_size for e in self._entries.values()) / total_entries if total_entries > 0 else 0

            return {
                'backend': 'json',
                'total_entries': total_entries,
                'avg_confidence': avg_confidence,
                'avg_sample_size': avg_sample_size,
                'file_path': str(self.file_path),
                'compressed': self.compress
            }

    def cleanup(self) -> bool:
        """Clean up resources."""
        return True


class CSVStorage(StorageBackend):
    """CSV file-based storage backend."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._lock = threading.RLock()
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._load_entries()

    def _load_entries(self):
        """Load entries from CSV file."""
        if not self.file_path.exists():
            return

        try:
            with self._lock:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Parse nested JSON fields
                            market_condition = json.loads(row['market_condition'])
                            strategy_metadata = json.loads(row['strategy_metadata'])
                            performance = json.loads(row['performance'])

                            entry_data = {
                                'id': row['id'],
                                'market_condition': market_condition,
                                'strategy_metadata': strategy_metadata,
                                'performance': performance,
                                'outcome': row['outcome'],
                                'confidence_score': float(row['confidence_score']),
                                'sample_size': int(row['sample_size']),
                                'last_updated': row['last_updated'],
                                'tags': json.loads(row['tags']) if row['tags'] else [],
                                'notes': row['notes'] if row['notes'] else None
                            }

                            entry = KnowledgeEntry.from_dict(entry_data)
                            self._entries[entry.id] = entry
                        except Exception as e:
                            print(f"Failed to load entry {row.get('id', 'unknown')}: {e}")

        except (csv.Error, FileNotFoundError) as e:
            print(f"Failed to load CSV knowledge base: {e}")
            self._entries = {}

    def _save_entries(self):
        """Save entries to CSV file."""
        try:
            with self._lock:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = [
                        'id', 'market_condition', 'strategy_metadata', 'performance',
                        'outcome', 'confidence_score', 'sample_size', 'last_updated',
                        'tags', 'notes'
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for entry in self._entries.values():
                        row = {
                            'id': entry.id,
                            'market_condition': json.dumps(entry.market_condition.to_dict()),
                            'strategy_metadata': json.dumps(entry.strategy_metadata.to_dict()),
                            'performance': json.dumps(entry.performance.to_dict()),
                            'outcome': entry.outcome.value,
                            'confidence_score': entry.confidence_score,
                            'sample_size': entry.sample_size,
                            'last_updated': entry.last_updated.isoformat(),
                            'tags': json.dumps(entry.tags),
                            'notes': entry.notes
                        }
                        writer.writerow(row)

        except Exception as e:
            print(f"Failed to save CSV knowledge base: {e}")
            return False
        return True

    def save_entry(self, entry: KnowledgeEntry) -> bool:
        """Save a knowledge entry."""
        errors = validate_knowledge_entry(entry)
        if errors:
            print(f"Invalid entry: {errors}")
            return False

        with self._lock:
            self._entries[entry.id] = entry
            return self._save_entries()

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a knowledge entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def query_entries(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """Query knowledge entries based on criteria."""
        import time
        start_time = time.time()

        with self._lock:
            candidates = list(self._entries.values())

        # Apply filters (same as JSON storage)
        filtered = []
        for entry in candidates:
            if query.market_regime and entry.market_condition.regime != query.market_regime:
                continue
            if query.strategy_name and entry.strategy_metadata.name != query.strategy_name:
                continue
            if query.strategy_category and entry.strategy_metadata.category != query.strategy_category:
                continue
            if entry.confidence_score < query.min_confidence:
                continue
            if entry.sample_size < query.min_sample_size:
                continue
            if query.timeframe and entry.strategy_metadata.timeframe != query.timeframe:
                continue
            if query.tags:
                if not any(tag in entry.tags for tag in query.tags):
                    continue

            filtered.append(entry)

        # Sort results
        if query.sort_by == "confidence_score":
            filtered.sort(key=lambda x: x.confidence_score, reverse=True)
        elif query.sort_by == "win_rate":
            filtered.sort(key=lambda x: x.performance.win_rate, reverse=True)
        elif query.sort_by == "profit_factor":
            filtered.sort(key=lambda x: x.performance.profit_factor, reverse=True)
        elif query.sort_by == "sample_size":
            filtered.sort(key=lambda x: x.sample_size, reverse=True)

        # Apply limit
        results = filtered[:query.limit]

        execution_time = time.time() - start_time
        return KnowledgeQueryResult(results, len(filtered), query, execution_time)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                return self._save_entries()
        return False

    def list_entries(self, limit: int = 100) -> List[KnowledgeEntry]:
        """List all knowledge entries."""
        with self._lock:
            entries = list(self._entries.values())
            return entries[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_entries = len(self._entries)
            avg_confidence = sum(e.confidence_score for e in self._entries.values()) / total_entries if total_entries > 0 else 0
            avg_sample_size = sum(e.sample_size for e in self._entries.values()) / total_entries if total_entries > 0 else 0

            return {
                'backend': 'csv',
                'total_entries': total_entries,
                'avg_confidence': avg_confidence,
                'avg_sample_size': avg_sample_size,
                'file_path': str(self.file_path)
            }

    def cleanup(self) -> bool:
        """Clean up resources."""
        return True


class SQLiteStorage(StorageBackend):
    """SQLite database storage backend."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_entries (
                        id TEXT PRIMARY KEY,
                        market_condition TEXT NOT NULL,
                        strategy_metadata TEXT NOT NULL,
                        performance TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        sample_size INTEGER NOT NULL,
                        last_updated TEXT NOT NULL,
                        tags TEXT,
                        notes TEXT
                    )
                ''')

                # Create indexes for better query performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_regime ON knowledge_entries(json_extract(market_condition, "$.regime"))')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_name ON knowledge_entries(json_extract(strategy_metadata, "$.name"))')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON knowledge_entries(confidence_score)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sample_size ON knowledge_entries(sample_size)')

                conn.commit()
            finally:
                conn.close()

    def save_entry(self, entry: KnowledgeEntry) -> bool:
        """Save a knowledge entry."""
        errors = validate_knowledge_entry(entry)
        if errors:
            print(f"Invalid entry: {errors}")
            return False

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO knowledge_entries
                    (id, market_condition, strategy_metadata, performance, outcome,
                     confidence_score, sample_size, last_updated, tags, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.id,
                    json.dumps(entry.market_condition.to_dict()),
                    json.dumps(entry.strategy_metadata.to_dict()),
                    json.dumps(entry.performance.to_dict()),
                    entry.outcome.value,
                    entry.confidence_score,
                    entry.sample_size,
                    entry.last_updated.isoformat(),
                    json.dumps(entry.tags),
                    entry.notes
                ))
                conn.commit()
                return True
            except Exception as e:
                print(f"Failed to save entry to SQLite: {e}")
                return False
            finally:
                conn.close()

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a knowledge entry by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute('SELECT * FROM knowledge_entries WHERE id = ?', (entry_id,))
                row = cursor.fetchone()
                if row:
                    entry_data = {
                        'id': row[0],
                        'market_condition': json.loads(row[1]),
                        'strategy_metadata': json.loads(row[2]),
                        'performance': json.loads(row[3]),
                        'outcome': row[4],
                        'confidence_score': row[5],
                        'sample_size': row[6],
                        'last_updated': row[7],
                        'tags': json.loads(row[8]) if row[8] else [],
                        'notes': row[9]
                    }
                    return KnowledgeEntry.from_dict(entry_data)
                return None
            except Exception as e:
                print(f"Failed to get entry from SQLite: {e}")
                return None
            finally:
                conn.close()

    def query_entries(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """Query knowledge entries based on criteria."""
        import time
        start_time = time.time()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Build query conditions
                conditions = []
                params = []

                if query.market_regime:
                    conditions.append("json_extract(market_condition, '$.regime') = ?")
                    params.append(query.market_regime.value)

                if query.strategy_name:
                    conditions.append("json_extract(strategy_metadata, '$.name') = ?")
                    params.append(query.strategy_name)

                if query.strategy_category:
                    conditions.append("json_extract(strategy_metadata, '$.category') = ?")
                    params.append(query.strategy_category.value)

                if query.min_confidence > 0:
                    conditions.append("confidence_score >= ?")
                    params.append(query.min_confidence)

                if query.min_sample_size > 1:
                    conditions.append("sample_size >= ?")
                    params.append(query.min_sample_size)

                if query.timeframe:
                    conditions.append("json_extract(strategy_metadata, '$.timeframe') = ?")
                    params.append(query.timeframe)

                # Build ORDER BY clause
                order_by = "confidence_score DESC"
                if query.sort_by == "win_rate":
                    order_by = "json_extract(performance, '$.win_rate') DESC"
                elif query.sort_by == "profit_factor":
                    order_by = "json_extract(performance, '$.profit_factor') DESC"
                elif query.sort_by == "sample_size":
                    order_by = "sample_size DESC"

                # Build final query
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                sql = f"SELECT * FROM knowledge_entries WHERE {where_clause} ORDER BY {order_by} LIMIT ?"
                params.append(query.limit)

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                # Convert rows to KnowledgeEntry objects
                entries = []
                for row in rows:
                    try:
                        entry_data = {
                            'id': row[0],
                            'market_condition': json.loads(row[1]),
                            'strategy_metadata': json.loads(row[2]),
                            'performance': json.loads(row[3]),
                            'outcome': row[4],
                            'confidence_score': row[5],
                            'sample_size': row[6],
                            'last_updated': row[7],
                            'tags': json.loads(row[8]) if row[8] else [],
                            'notes': row[9]
                        }
                        entry = KnowledgeEntry.from_dict(entry_data)
                        entries.append(entry)
                    except Exception as e:
                        print(f"Failed to parse entry {row[0]}: {e}")

                # Get total count for pagination info
                count_sql = f"SELECT COUNT(*) FROM knowledge_entries WHERE {where_clause}"
                count_params = params[:-1]  # Remove limit parameter
                cursor = conn.execute(count_sql, count_params)
                total_found = cursor.fetchone()[0]

                execution_time = time.time() - start_time
                return KnowledgeQueryResult(entries, total_found, query, execution_time)

            except Exception as e:
                print(f"Failed to query SQLite: {e}")
                execution_time = time.time() - start_time
                return KnowledgeQueryResult([], 0, query, execution_time)
            finally:
                conn.close()

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute('DELETE FROM knowledge_entries WHERE id = ?', (entry_id,))
                conn.commit()
                return True
            except Exception as e:
                print(f"Failed to delete entry from SQLite: {e}")
                return False
            finally:
                conn.close()

    def list_entries(self, limit: int = 100) -> List[KnowledgeEntry]:
        """List all knowledge entries."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute('SELECT * FROM knowledge_entries LIMIT ?', (limit,))
                rows = cursor.fetchall()

                entries = []
                for row in rows:
                    try:
                        entry_data = {
                            'id': row[0],
                            'market_condition': json.loads(row[1]),
                            'strategy_metadata': json.loads(row[2]),
                            'performance': json.loads(row[3]),
                            'outcome': row[4],
                            'confidence_score': row[5],
                            'sample_size': row[6],
                            'last_updated': row[7],
                            'tags': json.loads(row[8]) if row[8] else [],
                            'notes': row[9]
                        }
                        entry = KnowledgeEntry.from_dict(entry_data)
                        entries.append(entry)
                    except Exception as e:
                        print(f"Failed to parse entry {row[0]}: {e}")

                return entries
            except Exception as e:
                print(f"Failed to list entries from SQLite: {e}")
                return []
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Get total entries
                cursor = conn.execute('SELECT COUNT(*) FROM knowledge_entries')
                total_entries = cursor.fetchone()[0]

                # Get average confidence and sample size
                cursor = conn.execute('SELECT AVG(confidence_score), AVG(sample_size) FROM knowledge_entries')
                avg_confidence, avg_sample_size = cursor.fetchone()

                return {
                    'backend': 'sqlite',
                    'total_entries': total_entries,
                    'avg_confidence': avg_confidence or 0,
                    'avg_sample_size': avg_sample_size or 0,
                    'db_path': str(self.db_path)
                }
            except Exception as e:
                print(f"Failed to get SQLite stats: {e}")
                return {
                    'backend': 'sqlite',
                    'total_entries': 0,
                    'avg_confidence': 0,
                    'avg_sample_size': 0,
                    'db_path': str(self.db_path)
                }
            finally:
                conn.close()

    def cleanup(self) -> bool:
        """Clean up resources."""
        return True


class KnowledgeStorage:
    """Main knowledge storage manager supporting multiple backends."""

    def __init__(self, backend: str = 'json', **kwargs):
        """
        Initialize knowledge storage.

        Args:
            backend: Storage backend ('json', 'csv', 'sqlite')
            **kwargs: Backend-specific parameters
        """
        self.backend_type = backend
        self.backend = self._create_backend(backend, **kwargs)

    def _create_backend(self, backend: str, **kwargs) -> StorageBackend:
        """Create storage backend instance."""
        if backend == 'json':
            file_path = kwargs.get('file_path', 'knowledge_base/knowledge.json')
            compress = kwargs.get('compress', False)
            return JSONStorage(file_path, compress)
        elif backend == 'csv':
            file_path = kwargs.get('file_path', 'knowledge_base/knowledge.csv')
            return CSVStorage(file_path)
        elif backend == 'sqlite':
            db_path = kwargs.get('db_path', 'knowledge_base/knowledge.db')
            return SQLiteStorage(db_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def save_entry(self, entry: KnowledgeEntry) -> bool:
        """Save a knowledge entry."""
        return self.backend.save_entry(entry)

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a knowledge entry by ID."""
        return self.backend.get_entry(entry_id)

    def query_entries(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """Query knowledge entries based on criteria."""
        return self.backend.query_entries(query)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        return self.backend.delete_entry(entry_id)

    def list_entries(self, limit: int = 100) -> List[KnowledgeEntry]:
        """List all knowledge entries."""
        return self.backend.list_entries(limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.backend.get_stats()

    def cleanup(self) -> bool:
        """Clean up resources."""
        return self.backend.cleanup()

    def migrate_backend(self, new_backend: str, **kwargs) -> bool:
        """Migrate knowledge base to a different backend."""
        try:
            # Get all entries from current backend
            all_entries = self.list_entries(limit=10000)  # Reasonable limit

            # Create new backend
            new_storage = KnowledgeStorage(new_backend, **kwargs)

            # Save all entries to new backend
            success_count = 0
            for entry in all_entries:
                if new_storage.save_entry(entry):
                    success_count += 1

            print(f"Migrated {success_count}/{len(all_entries)} entries to {new_backend}")
            return success_count == len(all_entries)

        except Exception as e:
            print(f"Failed to migrate backend: {e}")
            return False
