"""
autonomous_learner.py — Turkish-AI Otonom Öğrenme Motoru v1

Yapay zeka arka planda sürekli:
  • Web sitelerinden veri çeker (RSS + scraping)
  • Çektiği veriyle online öğrenme yapar
  • Konuşmalardan öğrenir
  • Hafızasını günceller
  • Node.js tarafındaki /search endpoint'inden haber çeker

Kullanım (server_api.py içinde):
  from autonomous_learner import AutonomousLearner
  learner = AutonomousLearner(ai)
  learner.start()
"""

import threading
import time
import json
import os
import sys
import random
import hashlib
import requests
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


# ── Haber/İçerik Kaynakları ───────────────────────────────
RSS_FEEDS = [
    {"url": "https://www.sabah.com.tr/rss/anasayfa.xml",           "name": "Sabah",      "category": "gündem"},
    {"url": "https://www.haberturk.com/rss/tk/anasayfa.xml",       "name": "Habertürk",  "category": "gündem"},
    {"url": "https://www.milliyet.com.tr/rss/rssNew/gundemRss.xml", "name": "Milliyet",   "category": "gündem"},
    {"url": "https://tr.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=5&format=json",
                                                                    "name": "Wikipedia",  "category": "bilgi"},
    {"url": "https://feeds.bbci.co.uk/turkish/rss.xml",            "name": "BBC Türkçe", "category": "dünya"},
]

WIKIPEDIA_API = "https://tr.wikipedia.org/w/api.php"

# ── Öğrenme durumu dosyası ─────────────────────────────────
LEARNER_STATE_FILE = os.path.join(
    os.path.dirname(__file__), "data", "learner_state.json"
)


class AutonomousLearner:
    """
    Turkish-AI'nin otonom öğrenme motoru.
    Arka planda thread olarak çalışır, ana süreci bloklamaz.
    """

    def __init__(self, ai, node_server_url: str = "http://localhost:3000"):
        self.ai              = ai
        self.node_url        = node_server_url
        self._running        = False
        self._thread         = None
        self._seen_hashes    = deque(maxlen=5000)  # Sıralı, otomatik eski siler
        self._seen_hashes_set = set()              # Hızlı lookup için
        self._learn_count    = 0
        self._last_web_fetch = 0
        self._last_wiki      = 0
        self._cycle_count    = 0

        # Durum yükle
        self._state = self._load_state()
        loaded_hashes = self._state.get("seen_hashes", [])
        self._seen_hashes = deque(loaded_hashes, maxlen=5000)
        self._seen_hashes_set = set(loaded_hashes)
        self._learn_count = self._state.get("learn_count", 0)
        self._cycle_count = self._state.get("cycle_count", 0)

        print(f"[AutonomousLearner] 🤖 Motor hazır — {self._learn_count} öğrenme yapıldı")

    # ──────────────────────────────────────────────────────
    # BAŞLAT / DURDUR
    # ──────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="AutonomousLearner")
        self._thread.start()
        print("[AutonomousLearner] ▶️  Otonom öğrenme başladı (arka plan)")

    def stop(self):
        self._running = False
        print("[AutonomousLearner] ⏹️  Otonom öğrenme durduruldu")

    # ──────────────────────────────────────────────────────
    # ANA DÖNGÜ
    # ──────────────────────────────────────────────────────

    def _loop(self):
        """Her cycle'da farklı bir öğrenme görevi çalıştırır."""
        # İlk başlamada 10 saniye bekle (model yüklensin)
        time.sleep(10)

        while self._running:
            try:
                self._cycle_count += 1
                now = time.time()

                # Her 5 cycle'da bir RSS haber çek (yaklaşık 25 dakika)
                if self._cycle_count % 5 == 0:
                    self._fetch_and_learn_rss()

                # Her 3 cycle'da bir Wikipedia çek (yaklaşık 15 dakika)
                if self._cycle_count % 3 == 0:
                    self._fetch_and_learn_wikipedia()

                # Her cycle'da Node.js'ten haber çek
                self._fetch_from_node_search()

                # Durumu kaydet
                self._save_state()

                print(f"[AutonomousLearner] 🔄 Cycle {self._cycle_count} tamamlandı | Toplam öğrenme: {self._learn_count}")

            except Exception as e:
                print(f"[AutonomousLearner] ❌ Döngü hatası: {e}")

            # 5 dakika bekle (M4'ü yormamak için)
            time.sleep(300)

    # ──────────────────────────────────────────────────────
    # RSS HABER ÇEK + ÖĞREN
    # ──────────────────────────────────────────────────────

    def _fetch_and_learn_rss(self):
        """RSS feed'lerden haber çekip öğrenir."""
        import re

        feed = random.choice([f for f in RSS_FEEDS if f["name"] != "Wikipedia"])
        print(f"[AutonomousLearner] 📰 RSS çekiliyor: {feed['name']}")

        try:
            r = requests.get(
                feed["url"],
                timeout=10,
                headers={"User-Agent": "TurkishAI-Learner/1.0"}
            )
            xml = r.text

            # Item'ları parse et
            items = re.findall(r"<item>([\s\S]*?)</item>", xml)
            texts = []

            for item in items[:10]:
                title_m = re.search(r"<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?</title>", item)
                desc_m  = re.search(r"<description>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?</description>", item)

                title = self._strip_tags(title_m.group(1)).strip() if title_m else ""
                desc  = self._strip_tags(desc_m.group(1)).strip()  if desc_m  else ""

                if title and len(title) > 10:
                    text = f"{title}. {desc}" if desc else title
                    texts.append(text[:500])

            if texts:
                self._learn_texts(texts, source=feed["name"])
                print(f"[AutonomousLearner] ✅ {feed['name']} → {len(texts)} haber öğrenildi")

        except Exception as e:
            print(f"[AutonomousLearner] ⚠️ RSS hatası ({feed['name']}): {e}")

    # ──────────────────────────────────────────────────────
    # WİKİPEDİA ÇEK + ÖĞREN
    # ──────────────────────────────────────────────────────

    def _fetch_and_learn_wikipedia(self):
        """Wikipedia'dan rastgele makale çekip öğrenir."""
        print("[AutonomousLearner] 📚 Wikipedia çekiliyor...")

        try:
            # Rastgele makale ID'leri al
            r = requests.get(
                WIKIPEDIA_API,
                params={
                    "action": "query",
                    "list":   "random",
                    "rnnamespace": 0,
                    "rnlimit": 3,
                    "format": "json"
                },
                timeout=10,
                headers={"User-Agent": "TurkishAI-Learner/1.0"}
            )
            data    = r.json()
            pages   = data.get("query", {}).get("random", [])
            page_ids = [str(p["id"]) for p in pages]

            if not page_ids:
                return

            # Makale içeriklerini çek
            r2 = requests.get(
                WIKIPEDIA_API,
                params={
                    "action":      "query",
                    "pageids":     "|".join(page_ids),
                    "prop":        "extracts",
                    "exintro":     True,
                    "explaintext": True,
                    "exsentences": 5,
                    "format":      "json"
                },
                timeout=10,
                headers={"User-Agent": "TurkishAI-Learner/1.0"}
            )
            data2 = r2.json()
            pages2 = data2.get("query", {}).get("pages", {})

            texts = []
            for page in pages2.values():
                extract = page.get("extract", "")
                title   = page.get("title", "")
                if extract and len(extract) > 50:
                    text = f"{title}: {extract[:600]}"
                    texts.append(text)

            if texts:
                self._learn_texts(texts, source="Wikipedia")
                print(f"[AutonomousLearner] ✅ Wikipedia → {len(texts)} makale öğrenildi")

        except Exception as e:
            print(f"[AutonomousLearner] ⚠️ Wikipedia hatası: {e}")

    # ──────────────────────────────────────────────────────
    # NODE.JS SEARCH'TEN ÇEK
    # ──────────────────────────────────────────────────────

    def _fetch_from_node_search(self):
        """Node.js /search endpoint'inden güncel haberler çeker."""
        topics = [
            "Türkiye son dakika",
            "teknoloji haberleri",
            "bilim gelişmeleri",
            "yapay zeka",
            "ekonomi haberleri",
        ]
        topic = random.choice(topics)

        try:
            r = requests.post(
                f"{self.node_url}/search",
                json={"query": topic, "maxResults": 3},
                timeout=10
            )
            data    = r.json()
            results = data.get("results", [])

            if not results:
                return

            texts = []
            for result in results:
                title   = result.get("title", "")
                snippet = result.get("snippet", "")
                if title:
                    text = f"{title}. {snippet}" if snippet else title
                    texts.append(text[:400])

            if texts:
                self._learn_texts(texts, source="WebSearch")
                print(f"[AutonomousLearner] 🌐 Node.js arama → '{topic}': {len(texts)} sonuç öğrenildi")

        except Exception as e:
            # Node.js kapalıysa sessizce geç
            pass

    # ──────────────────────────────────────────────────────
    # ONLINE ÖĞRENME
    # ──────────────────────────────────────────────────────

    def _learn_texts(self, texts: List[str], source: str = "web"):
        """Verilen metinleri modele öğretir (online learning)."""
        if not texts or not self.ai or not self.ai.is_initialized:
            return

        new_texts = []
        for text in texts:
            if not text or len(text) < 20:
                continue
            # Hash ile duplicate kontrolü
            h = hashlib.md5(text.encode()).hexdigest()
            if h in self._seen_hashes_set:
                continue
            # deque maxlen=5000 olduğu için taşınca otomatik en eskiyi siler
            if len(self._seen_hashes) == self._seen_hashes.maxlen:
                oldest = self._seen_hashes[0]
                self._seen_hashes_set.discard(oldest)
            self._seen_hashes.append(h)
            self._seen_hashes_set.add(h)
            new_texts.append(text)

        if not new_texts:
            return

        try:
            # Online learner varsa kullan
            if self.ai.online_learner:
                for text in new_texts[:5]:  # Her seferinde max 5 metin
                    # Metni soru-cevap formatına çevir
                    import re as _re
                    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]
                    if len(sentences) >= 2:
                        prompt   = sentences[0] + " hakkında ne biliyorsun?"
                        response = ". ".join(sentences[1:3])
                        self.ai.online_learner.learn_from_interaction(
                            prompt, response, feedback=1.0
                        )
                        self._learn_count += 1
                    elif sentences:
                        # Tek cümle — tamamlama olarak öğret
                        self.ai.online_learner.learn_from_interaction(
                            sentences[0], text[:200], feedback=1.0
                        )
                        self._learn_count += 1

            # Hafızaya da kaydet (hızlı erişim için)
            if self.ai.memory:
                for text in new_texts[:3]:
                    key = f"learned:{source}:{datetime.now().strftime('%Y%m%d_%H%M')}:{text[:20]}"
                    self.ai.memory.remember_user_fact(
                        f"[{source}] {text[:200]}", category="auto_learned"
                    )

        except Exception as e:
            print(f"[AutonomousLearner] ⚠️ Öğrenme hatası: {e}")

    # ──────────────────────────────────────────────────────
    # KONUŞMADAN ÖĞREN (dışarıdan çağrılır)
    # ──────────────────────────────────────────────────────

    def learn_from_conversation(self, user_msg: str, ai_response: str):
        """Her konuşmadan otomatik öğrenir."""
        if not user_msg or not ai_response:
            return
        if len(user_msg) < 5 or len(ai_response) < 5:
            return

        try:
            if self.ai.online_learner:
                self.ai.online_learner.learn_from_interaction(
                    user_msg, ai_response, feedback=1.0
                )
                self._learn_count += 1
        except Exception as e:
            pass

    # ──────────────────────────────────────────────────────
    # DURUM
    # ──────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "running":      self._running,
            "learn_count":  self._learn_count,
            "cycle_count":  self._cycle_count,
            "seen_hashes":  len(self._seen_hashes),
            "last_updated": datetime.now().isoformat(),
        }

    # ──────────────────────────────────────────────────────
    # YARDIMCILAR
    # ──────────────────────────────────────────────────────

    def _strip_tags(self, text: str) -> str:
        import re
        return re.sub(r"<[^>]+>", " ", text).strip()

    def _load_state(self) -> Dict:
        os.makedirs(os.path.dirname(LEARNER_STATE_FILE), exist_ok=True)
        try:
            if os.path.exists(LEARNER_STATE_FILE):
                with open(LEARNER_STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"seen_hashes": [], "learn_count": 0}

    def _save_state(self):
        try:
            state = {
                "seen_hashes": list(self._seen_hashes)[-2000:],  # Son 2000'i sakla
                "learn_count": self._learn_count,
                "cycle_count": self._cycle_count,
                "last_saved":  datetime.now().isoformat(),
            }
            with open(LEARNER_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass
