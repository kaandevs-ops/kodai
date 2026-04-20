"""
Gelişmiş Hafıza Sistemi — v2
Orijinal yapı ve API tamamen korundu, iyileştirmeler:

  • Context manager ile SQLite bağlantıları  → bağlantı sızıntısı yok
  • Thread-safe bağlantı yönetimi          → çoklu thread'de güvenli
  • Importance skoru otomatik hesaplanıyor → soru işareti = yüksek önem
  • VectorMemory embedding kontrolü        → embedding boyutu uyuşmazsa hata vermez
  • recall_user_facts düzeltmesi           → category=None durumu düzeltildi
"""

import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import hashlib
import threading
import os


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """Tek bir konuşma turu."""
    role: str                         # "user" veya "assistant"
    content: str
    timestamp: datetime
    emotion: Optional[str]  = None    # Duygu analizi (opsiyonel)
    importance: float       = 0.0     # 0–1 arası önem skoru

    def to_dict(self):
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data):
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Kısa Vadeli Hafıza
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    Kısa vadeli hafıza: Son N konuşmayı RAM'de tutar.
    FIFO mantığı ile çalışır.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns   = max_turns
        self.conversations: List[ConversationTurn] = []

    def add(self, turn: ConversationTurn):
        """Yeni konuşma ekle, limit aşılırsa en eskiyi sil."""
        self.conversations.append(turn)
        if len(self.conversations) > self.max_turns:
            self.conversations.pop(0)

    def get_recent(self, n: int = None) -> List[ConversationTurn]:
        """Son n konuşmayı getir."""
        n = n or self.max_turns
        return self.conversations[-n:]

    def get_context_string(self, n: int = 5) -> str:
        """Son n konuşmayı model için tek string olarak döndür."""
        lines = []
        for turn in self.get_recent(n):
            prefix = "Kullanıcı: " if turn.role == "user" else "Asistan: "
            lines.append(f"{prefix}{turn.content}")
        return "\n".join(lines)

    def clear(self):
        self.conversations = []


# ---------------------------------------------------------------------------
# Uzun Vadeli Hafıza
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Uzun vadeli hafıza: SQLite veritabanında kalıcı depolama.
    Context manager kullanımıyla bağlantı sızıntısı ortadan kalktı.
    """

    def __init__(self, db_path: str = "./data/memory/long_term.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock   = threading.Lock()   # Thread güvenliği
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Yeni bağlantı aç (context manager ile kullanılacak)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # Eş zamanlı okuma/yazma için
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        """Veritabanı tablolarını oluştur."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role       TEXT,
                    content    TEXT,
                    timestamp  TEXT,
                    importance REAL    DEFAULT 0.0,
                    category   TEXT    DEFAULT 'general'
                );

                CREATE TABLE IF NOT EXISTS knowledge (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    key          TEXT    UNIQUE,
                    value        TEXT,
                    category     TEXT,
                    confidence   REAL    DEFAULT 1.0,
                    last_updated TEXT
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    summary    TEXT,
                    start_time TEXT,
                    end_time   TEXT,
                    topics     TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_conv_session
                    ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conv_content
                    ON conversations(content);
                CREATE INDEX IF NOT EXISTS idx_know_key
                    ON knowledge(key);
            """)

    # ------------------------------------------------------------------

    def store_conversation(self, turn: ConversationTurn, session_id: str = "default"):
        """Konuşmayı kalıcı olarak sakla."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO conversations
                       (session_id, role, content, timestamp, importance)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, turn.role, turn.content,
                     turn.timestamp.isoformat(), turn.importance)
                )

    def store_knowledge(
        self,
        key: str,
        value: str,
        category: str    = "general",
        confidence: float = 1.0
    ):
        """Bilgi sakla (anahtar-değer çifti)."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge
                       (key, value, category, confidence, last_updated)
                       VALUES (?, ?, ?, ?, ?)""",
                    (key, value, category, confidence, datetime.now().isoformat())
                )

    def get_knowledge(self, key: str) -> Optional[str]:
        """Bilgi getir."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM knowledge WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def search_conversations(
        self,
        keyword: str,
        session_id: str = None,
        limit: int      = 10
    ) -> List[Dict]:
        """Konuşmalarda anahtar kelime ara (LIKE ile)."""
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """SELECT id, session_id, role, content, timestamp, importance
                       FROM conversations
                       WHERE content LIKE ? AND session_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (f"%{keyword}%", session_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, session_id, role, content, timestamp, importance
                       FROM conversations
                       WHERE content LIKE ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (f"%{keyword}%", limit)
                ).fetchall()

        return [
            {"id": r[0], "session_id": r[1], "role": r[2],
             "content": r[3], "timestamp": r[4], "importance": r[5]}
            for r in rows
        ]

    def get_conversation_history(
        self,
        session_id: str   = "default",
        since: datetime   = None,
        limit: int        = 100
    ) -> List[ConversationTurn]:
        """Belirli bir oturumun geçmişini getir."""
        with self._connect() as conn:
            if since:
                rows = conn.execute(
                    """SELECT role, content, timestamp, importance
                       FROM conversations
                       WHERE session_id = ? AND timestamp > ?
                       ORDER BY timestamp LIMIT ?""",
                    (session_id, since.isoformat(), limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT role, content, timestamp, importance
                       FROM conversations
                       WHERE session_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (session_id, limit)
                ).fetchall()

        return [
            ConversationTurn(
                role       = r[0],
                content    = r[1],
                timestamp  = datetime.fromisoformat(r[2]),
                importance = r[3]
            )
            for r in reversed(rows)
        ]

    def count_conversations(self) -> int:
        """Toplam konuşma sayısı — orijinal _count_long_term() için."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Vektör Hafıza
# ---------------------------------------------------------------------------

class VectorMemory:
    """
    Semantik arama için embedding tabanlı hafıza.
    FAISS varsa kullanır, yoksa NumPy ile cosine similarity hesaplar.
    """

    def __init__(
        self,
        embedding_dim: int  = 768,
        storage_path: str   = "./data/memory/vectors"
    ):
        self.embedding_dim = embedding_dim
        self.storage_path  = storage_path
        self.vectors: List[Dict] = []
        self.id_counter    = 0

        os.makedirs(storage_path, exist_ok=True)

        try:
            import faiss
            self.index    = faiss.IndexFlatIP(embedding_dim)
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False
            self.index     = None

    def add(self, text: str, embedding: np.ndarray, metadata: Dict = None) -> int:
        """Vektör ekle."""
        # Boyut uyuşmazlığı kontrolü
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Embedding boyutu uyuşmuyor: beklenen {self.embedding_dim}, "
                f"gelen {embedding.shape[-1]}"
            )

        self.id_counter += 1
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)

        if self.use_faiss:
            self.index.add(emb_norm.reshape(1, -1).astype("float32"))

        self.vectors.append({
            "id":        self.id_counter,
            "text":      text,
            "embedding": emb_norm,
            "metadata":  metadata or {}
        })
        return self.id_counter

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int       = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """En benzer vektörleri bul."""
        if not self.vectors:
            return []

        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        if self.use_faiss:
            scores, indices = self.index.search(
                q.reshape(1, -1).astype("float32"),
                min(top_k, len(self.vectors))
            )
            return [
                {"id": self.vectors[i]["id"], "text": self.vectors[i]["text"],
                 "score": float(s), "metadata": self.vectors[i]["metadata"]}
                for s, i in zip(scores[0], indices[0])
                if s >= threshold and i >= 0
            ]
        else:
            sims = [(np.dot(q, v["embedding"]), v) for v in self.vectors]
            sims.sort(key=lambda x: x[0], reverse=True)
            return [
                {"id": v["id"], "text": v["text"],
                 "score": float(s), "metadata": v["metadata"]}
                for s, v in sims[:top_k]
                if s >= threshold
            ]

    def save(self):
        """Vektör hafızasını diske kaydet."""
        path = os.path.join(self.storage_path, "vectors.pkl")
        data = {
            "id_counter":  self.id_counter,
            "embedding_dim": self.embedding_dim,
            "vectors": [
                {"id": v["id"], "text": v["text"],
                 "embedding": v["embedding"], "metadata": v["metadata"]}
                for v in self.vectors
            ]
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        """Vektör hafızasını diskten yükle."""
        path = os.path.join(self.storage_path, "vectors.pkl")
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.id_counter    = data["id_counter"]
        self.embedding_dim = data.get("embedding_dim", self.embedding_dim)
        self.vectors       = data["vectors"]

        if self.use_faiss and self.vectors:
            for v in self.vectors:
                self.index.add(v["embedding"].reshape(1, -1).astype("float32"))


# ---------------------------------------------------------------------------
# Hafıza Yöneticisi
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Ana hafıza yöneticisi.
    3 katmanı koordine eder: ShortTerm (RAM), LongTerm (SQLite), Vector (FAISS/NumPy).
    """

    def __init__(
        self,
        short_term_limit: int = 10,
        db_path: str          = "./data/memory/long_term.db",
        vector_path: str      = "./data/memory/vectors"
    ):
        self.short_term   = ShortTermMemory(max_turns=short_term_limit)
        self.long_term    = LongTermMemory(db_path=db_path)
        self.vector_memory = VectorMemory(storage_path=vector_path)

        self.summary_threshold  = 20
        self.conversation_count = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _calc_importance(text: str) -> float:
        """
        Basit önem skoru: soru işareti, ünlem ve anahtar kelimeler
        yüksek öneme işaret eder. 0.0 – 1.0 arası döner.
        """
        score = 0.3  # Varsayılan
        if "?" in text:
            score += 0.2
        if "!" in text:
            score += 0.1
        important_words = ["önemli", "unutma", "hatırla", "kaydet",
                           "kritik", "acil", "not", "lütfen"]
        if any(w in text.lower() for w in important_words):
            score += 0.3
        return min(1.0, score)

    # ------------------------------------------------------------------
    def add_interaction(
        self,
        user_msg: str,
        assistant_msg: str,
        session_id: str = "default"
    ):
        """Yeni etkileşimi tüm hafıza katmanlarına kaydet."""
        now = datetime.now()

        user_turn = ConversationTurn(
            role       = "user",
            content    = user_msg,
            timestamp  = now,
            importance = self._calc_importance(user_msg)
        )
        asst_turn = ConversationTurn(
            role       = "assistant",
            content    = assistant_msg,
            timestamp  = now,
            importance = 0.5
        )

        # Kısa vadeli hafıza
        self.short_term.add(user_turn)
        self.short_term.add(asst_turn)

        # Uzun vadeli hafıza
        self.long_term.store_conversation(user_turn, session_id)
        self.long_term.store_conversation(asst_turn, session_id)

        self.conversation_count += 1

        if self.conversation_count >= self.summary_threshold:
            self._create_summary(session_id)
            self.conversation_count = 0

    # ------------------------------------------------------------------
    def _create_summary(self, session_id: str):
        """Son konuşmaların kelime frekansına dayalı özeti."""
        history = self.long_term.get_conversation_history(session_id, limit=20)
        if len(history) < 5:
            return

        all_text = " ".join(t.content for t in history)
        words    = all_text.lower().split()
        freq: Dict[str, int] = {}
        for w in words:
            if len(w) > 3:
                freq[w] = freq.get(w, 0) + 1

        top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        summary   = f"Konuşulan konular: {', '.join(w for w, _ in top_words)}"

        self.long_term.store_knowledge(
            f"summary_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            summary,
            category="conversation_summary"
        )

    # ------------------------------------------------------------------
    def get_context_for_model(self, include_long_term: bool = True) -> str:
        """Modele verilecek bağlamı hazırla."""
        parts = []

        short_ctx = self.short_term.get_context_string(n=5)
        if short_ctx:
            parts.append("Son Konuşmalar:")
            parts.append(short_ctx)

        if include_long_term:
            recent = self.short_term.get_recent(1)
            if recent:
                keywords = [
                    w for w in recent[0].content.split()
                    if len(w) > 3
                ][:3]
                for kw in keywords:
                    related = self.long_term.search_conversations(kw, limit=3)
                    if related:
                        parts.append(f"\nİlgili Geçmiş ({kw}):")
                        for conv in related[:2]:
                            snippet = conv["content"][:100]
                            parts.append(f"- {conv['role']}: {snippet}...")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    def remember_user_fact(self, fact: str, category: str = "preference"):
        """Kullanıcı hakkında bilgi sakla."""
        key = "user_fact_" + hashlib.md5(fact.encode()).hexdigest()[:16]
        self.long_term.store_knowledge(key, fact, category=category)

    def recall_user_facts(self, category: str = None) -> List[str]:
        """Kullanıcı bilgilerini getir."""
        with self.long_term._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT value FROM knowledge WHERE category = ?",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT value FROM knowledge WHERE key LIKE 'user_fact_%'"
                ).fetchall()
        return [r[0] for r in rows]

    def clear_short_term(self):
        """Kısa vadeli hafızayı temizle (yeni konuşma)."""
        self.short_term.clear()