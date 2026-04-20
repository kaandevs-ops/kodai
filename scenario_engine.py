"""
scenario_engine.py — Turkish-AI Olasılık & Senaryo Motoru v1

Sıfır dış bağımlılık. Tamamen istatistik + kural tabanlı.
Ollama yok, internet yok, API yok.

Ne yapar:
  • Her konuşmayı kaydeder (episodik hafıza)
  • Geçmişten olasılık hesaplar (frekans analizi)
  • "Eğer X → Y olur, Y → Z olur" zinciri kurar
  • Kullandıkça doğruluğu artar

Kullanım (server_api.py'ye ekle):
  from scenario_engine import ScenarioEngine
  scenario = ScenarioEngine()
  
  # Konuşma kaydı
  scenario.record("toplantıya geç kaldım", "özür diledim, geçti", outcome="positive")
  
  # Senaryo analizi
  result = scenario.analyze("Yarın toplantıya geç kalırsam ne olur?")
"""

import json
import os
import re
import time
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "scenarios")

STOP_WORDS = {
    "bir","bu","şu","ve","ile","de","da","ki","ne","mi","nasıl","neden",
    "için","ama","çok","daha","en","gibi","var","yok","bana","sana","ben",
    "sen","onun","biz","her","hiç","olan","ama","fakat","lakin","ancak",
    "eğer","ise","iken","bile","ya","veya","hem","ne","ya da","yani"
}


class ScenarioEngine:
    """
    Tamamen istatistik tabanlı olasılık ve senaryo simülasyon motoru.
    Ollama'ya sıfır bağımlılık.
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._episodes   = self._load("episodes.json", [])
        self._patterns   = self._load("patterns.json", {
            "outcomes":    {},   # konu → {positive:N, negative:N, neutral:N}
            "chains":      {},   # durum → sonraki durum listesi
            "timePatterns":{},   # saat → konu frekansı
        })
        self._prev_kw    = []
        print(f"[ScenarioEngine] 🎯 Yüklendi — {len(self._episodes)} episode, {len(self._patterns['outcomes'])} konu")

    # ──────────────────────────────────────────────────────
    # YARDIMCILAR
    # ──────────────────────────────────────────────────────

    def _kw(self, text: str) -> List[str]:
        """Metinden anlamlı anahtar kelimeler çıkar."""
        words = re.sub(r"[^a-züçğışöıA-ZÜÇĞİŞÖI\s]", " ", text.lower()).split()
        return [w for w in words if len(w) > 2 and w not in STOP_WORDS][:8]

    def _topic_key(self, kw: List[str]) -> str:
        if not kw:
            return "genel"
        count = 3 if len(kw) >= 3 else len(kw)
        return "_".join(sorted(kw[:count]))

    def _is_positive(self, text: str) -> bool:
        return bool(re.search(
            r"tamam|oldu|hallettim|çözdüm|teşekkür|harika|güzel|iyi|başardım|mükemmel",
            text, re.I
        ))

    def _is_negative(self, text: str) -> bool:
        return bool(re.search(
            r"olmadı|hata|sorun|çalışmıyor|kötü|berbat|başaramadım|yanlış|fail|error",
            text, re.I
        ))

    def _load(self, filename: str, default):
        path = os.path.join(DATA_DIR, filename)
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return default

    def _save(self, filename: str, data):
        path = os.path.join(DATA_DIR, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────
    # KAYIT — Her konuşmadan öğren
    # ──────────────────────────────────────────────────────

    def record(self, situation: str, result: str = "", outcome: str = "neutral"):
        """
        Bir durumu ve sonucunu kaydet.
        outcome: "positive" | "negative" | "neutral"
        """
        if not situation:
            return

        kw        = self._kw(situation)
        topic_key = self._topic_key(kw)
        hour      = datetime.now().hour

        # Outcome'u metinden otomatik tespit et
        if outcome == "neutral":
            if self._is_positive(result):
                outcome = "positive"
            elif self._is_negative(result):
                outcome = "negative"

        # Episode kaydet
        episode = {
            "id":        int(time.time() * 1000),
            "situation": situation[:200],
            "result":    result[:200],
            "outcome":   outcome,
            "keywords":  kw,
            "topic_key": topic_key,
            "hour":      hour,
            "date":      datetime.now().strftime("%Y-%m-%d"),
            "ts":        datetime.now().isoformat(),
        }
        self._episodes.insert(0, episode)
        if len(self._episodes) > 1000:
            self._episodes = self._episodes[:800]

        # Outcome sayacını güncelle
        if topic_key not in self._patterns["outcomes"]:
            self._patterns["outcomes"][topic_key] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        self._patterns["outcomes"][topic_key][outcome] += 1
        self._patterns["outcomes"][topic_key]["total"] += 1

        # Zincir: önceki durum → bu durum
        if self._prev_kw:
            prev_key = self._topic_key(self._prev_kw)
            if prev_key not in self._patterns["chains"]:
                self._patterns["chains"][prev_key] = {}
            chain_targets = self._patterns["chains"][prev_key]
            chain_targets[topic_key] = chain_targets.get(topic_key, 0) + 1

        # Saat bazlı pattern
        hour_key = str(hour)
        if hour_key not in self._patterns["timePatterns"]:
            self._patterns["timePatterns"][hour_key] = {}
        for w in kw:
            tp = self._patterns["timePatterns"][hour_key]
            tp[w] = tp.get(w, 0) + 1

        self._prev_kw = kw
        self._save("episodes.json", self._episodes)
        self._save("patterns.json", self._patterns)

    # ──────────────────────────────────────────────────────
    # OLASILIK HESAPLA
    # ──────────────────────────────────────────────────────

    def probability(self, situation: str) -> Dict:
        """
        Verilen durum için geçmişe dayalı olasılık hesapla.
        Döner: {"positive": 0.6, "negative": 0.2, "neutral": 0.2, "confidence": 0.7, "evidence_count": 15}
        """
        kw        = self._kw(situation)
        topic_key = self._topic_key(kw)

        # Doğrudan eşleşme
        outcomes = self._patterns["outcomes"].get(topic_key, {})
        total    = outcomes.get("total", 0)

        # Benzer episodlar
        similar = self._find_similar(situation, top_n=10)
        sim_total = len(similar)

        if total == 0 and sim_total == 0:
            return {
                "positive":       0.5,
                "negative":       0.3,
                "neutral":        0.2,
                "confidence":     0.0,
                "evidence_count": 0,
                "message":        "Yeterli geçmiş veri yok, tahmin yapılamıyor"
            }

        # Doğrudan + benzer verileri birleştir
        pos = outcomes.get("positive", 0)
        neg = outcomes.get("negative", 0)
        neu = outcomes.get("neutral", 0)

        for ep in similar:
            if ep["outcome"] == "positive":  pos += 0.5
            elif ep["outcome"] == "negative": neg += 0.5
            else: neu += 0.5

        grand_total = pos + neg + neu
        if grand_total == 0:
            grand_total = 1

        confidence = min(1.0, (total + sim_total * 0.5) / 20.0)

        return {
            "positive":       round(pos / grand_total, 2),
            "negative":       round(neg / grand_total, 2),
            "neutral":        round(neu / grand_total, 2),
            "confidence":     round(confidence, 2),
            "evidence_count": total + sim_total,
            "topic_key":      topic_key,
        }

    # ──────────────────────────────────────────────────────
    # SENARYO SİMÜLASYONU — "Eğer X → Y → Z" zinciri
    # ──────────────────────────────────────────────────────

    def analyze(self, situation: str, depth: int = 2) -> Dict:
        """
        Verilen durum için senaryo analizi yap.
        depth: kaç adım ileriye baksın (1-3)

        Örnek çıktı:
        {
          "situation": "Toplantıya geç kalırsam",
          "scenarios": [
            {
              "id": "A",
              "title": "Olumlu sonuç",
              "probability": 40,
              "description": "Özür dilersin, geçersin",
              "sub_scenarios": [...]
            },
            ...
          ],
          "best_action": "Erken çık, %70 başarı şansı artar",
          "risk_level": "medium",
          "confidence": 0.6
        }
        """
        kw        = self._kw(situation)
        topic_key = self._topic_key(kw)
        prob      = self.probability(situation)
        similar   = self._find_similar(situation, top_n=5)

        # Senaryoları oluştur
        scenarios = self._build_scenarios(situation, prob, similar, depth)

        # En iyi aksiyon
        best_action = self._suggest_action(situation, prob, similar)

        # Risk seviyesi
        risk = "low"
        if prob["negative"] > 0.5:
            risk = "high"
        elif prob["negative"] > 0.3:
            risk = "medium"

        result = {
            "situation":   situation,
            "scenarios":   scenarios,
            "best_action": best_action,
            "risk_level":  risk,
            "confidence":  prob["confidence"],
            "probability": prob,
            "similar_past": [
                {
                    "situation": ep["situation"][:80],
                    "outcome":   ep["outcome"],
                    "date":      ep["date"]
                }
                for ep in similar[:3]
            ],
            "analyzed_at": datetime.now().isoformat(),
        }

        # Kaydet
        self.record(situation, str(scenarios), outcome="neutral")

        return result

    def chain(self, start: str, steps: List[str]) -> List[Dict]:
        """
        Zincirleme senaryo: "Eğer X olursa, sonra Y olursa, sonra Z olursa"

        Örnek:
          chain("iş kurmak istiyorum", ["sermaye bulmak zorundayım", "ortak bulmak gerekiyor"])
        """
        results = []
        current = start

        # İlk adım
        first = self.analyze(current, depth=1)
        results.append({
            "step":      0,
            "situation": current,
            "analysis":  first,
        })

        # Zinciri izle
        for i, step in enumerate(steps[:3]):
            current = f"{current} → {step}"
            analysis = self.analyze(current, depth=1)
            results.append({
                "step":      i + 1,
                "condition": step,
                "situation": current,
                "analysis":  analysis,
            })

        # Genel özet
        all_risks  = [r["analysis"]["risk_level"] for r in results]
        final_risk = "high" if "high" in all_risks else "medium" if "medium" in all_risks else "low"
        avg_conf   = sum(r["analysis"]["confidence"] for r in results) / len(results)

        return {
            "chain":       results,
            "total_steps": len(results),
            "final_risk":  final_risk,
            "avg_confidence": round(avg_conf, 2),
            "summary":     self._chain_summary(results),
        }

    # ──────────────────────────────────────────────────────
    # YARDIMCI FONKSİYONLAR
    # ──────────────────────────────────────────────────────

    def _find_similar(self, situation: str, top_n: int = 5) -> List[Dict]:
        """Geçmiş episodlarda benzer durumları bul."""
        if not self._episodes:
            return []

        query_kw = set(self._kw(situation))
        scored   = []

        for ep in self._episodes:
            ep_kw   = set(ep.get("keywords", []))
            overlap = len(query_kw & ep_kw)
            if overlap == 0:
                continue

            # Tazelik bonusu
            try:
                age_days = (datetime.now() - datetime.fromisoformat(ep["ts"])).days
                freshness = max(0.5, 1.0 - age_days * 0.01)
            except Exception:
                freshness = 0.7

            current_hour = datetime.now().hour
            hour_diff = abs(ep.get("hour", current_hour) - current_hour)
            hour_bonus = 1.3 if hour_diff <= 2 else 1.0
            score = overlap * freshness * hour_bonus
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_n]]

    def _build_scenarios(self, situation: str, prob: Dict, similar: List[Dict], depth: int) -> List[Dict]:
        """Olasılık verilerinden senaryo listesi oluştur."""
        scenarios = []

        pos_pct = round(prob["positive"] * 100)
        neg_pct = round(prob["negative"] * 100)
        neu_pct = round(prob["neutral"]  * 100)

        # Olumlu senaryo
        pos_examples = [ep for ep in similar if ep["outcome"] == "positive"]
        pos_desc = (pos_examples[0]["result"][:100] if pos_examples and not pos_examples[0]["result"].startswith("[{") and not pos_examples[0]["result"].startswith("{") else "İşler yolunda gidebilir")
        scenarios.append({
            "id":          "A",
            "title":       "Olumlu sonuç",
            "probability": pos_pct,
            "description": pos_desc,
            "sub_scenarios": self._sub_scenarios(situation, "positive", depth) if depth > 1 else [],
        })

        # Olumsuz senaryo
        neg_examples = [ep for ep in similar if ep["outcome"] == "negative"]
        neg_desc = (neg_examples[0]["result"][:100] if neg_examples and not neg_examples[0]["result"].startswith("[{") and not neg_examples[0]["result"].startswith("{") else "Sorunlar çıkabilir")
        scenarios.append({
            "id":          "B",
            "title":       "Olumsuz sonuç",
            "probability": neg_pct,
            "description": neg_desc,
            "sub_scenarios": self._sub_scenarios(situation, "negative", depth) if depth > 1 else [],
        })

        # Nötr senaryo
        if neu_pct > 10:
            scenarios.append({
                "id":          "C",
                "title":       "Belirsiz sonuç",
                "probability": neu_pct,
                "description": "Durum netleşmeden sonuç söylemek zor",
                "sub_scenarios": [],
            })

        # Olasılığa göre sırala
        scenarios.sort(key=lambda x: x["probability"], reverse=True)
        return scenarios

    def _sub_scenarios(self, situation: str, parent_outcome: str, depth: int) -> List[Dict]:
        """Alt senaryo zinciri — bir sonraki adım ne olabilir."""
        if depth <= 1:
            return []

        kw        = self._kw(situation)
        topic_key = self._topic_key(kw)
        chain_map = self._patterns["chains"].get(topic_key, {})

        if not chain_map:
            return []

        # En sık görülen sonraki durumlar
        sorted_chains = sorted(chain_map.items(), key=lambda x: x[1], reverse=True)[:2]
        subs = []

        for next_topic, count in sorted_chains:
            next_outcomes = self._patterns["outcomes"].get(next_topic, {})
            next_total    = next_outcomes.get("total", 1)
            next_pos      = next_outcomes.get("positive", 0)
            subs.append({
                "id":          next_topic,
                "description": f"Sonra '{next_topic}' durumu oluşabilir",
                "probability": round((next_pos / next_total) * 100) if next_total > 0 else 50,
                "count":       count,
            })

        return subs

    def _suggest_action(self, situation: str, prob: Dict, similar: List[Dict]) -> str:
        """En iyi aksiyon öner."""
        pos_pct = round(prob["positive"] * 100)
        neg_pct = round(prob["negative"] * 100)
        conf    = prob["confidence"]

        if conf < 0.2:
            return "Yeterli geçmiş veri yok. Dikkatli ol ve sonucu kaydet."

        if neg_pct > 60:
            # Başarılı örneklerden öğren
            pos_examples = [ep for ep in similar if ep["outcome"] == "positive"]
            if pos_examples:
                return f"Geçmişte başarılı olan yaklaşım: {pos_examples[0]['situation'][:80]}"
            return f"Risk yüksek (%{neg_pct}). Alternatif bir yol düşün veya yardım al."

        if pos_pct > 60:
            return f"Başarı şansı yüksek (%{pos_pct}). Devam et."

        return f"Karışık sonuçlar (olumlu: %{pos_pct}, olumsuz: %{neg_pct}). Küçük adımlarla ilerle."

    def _chain_summary(self, chain_results: List[Dict]) -> str:
        """Zincir analizi için kısa özet."""
        risks     = [r["analysis"]["risk_level"] for r in chain_results]
        high_risk = risks.count("high")
        if high_risk > 1:
            return f"{len(chain_results)} adımlı zincirde {high_risk} yüksek riskli durum var. Dikkatli planla."
        elif "high" in risks:
            step = risks.index("high")
            return f"Adım {step + 1}'de yüksek risk var. O aşamaya dikkat et."
        return f"{len(chain_results)} adımlı zincir analizi tamamlandı. Genel risk: düşük-orta."

    # ──────────────────────────────────────────────────────
    # DURUM
    # ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        total    = len(self._episodes)
        positive = sum(1 for ep in self._episodes if ep.get("outcome") == "positive")
        negative = sum(1 for ep in self._episodes if ep.get("outcome") == "negative")

        top_topics = sorted(
            self._patterns["outcomes"].items(),
            key=lambda x: x[1].get("total", 0),
            reverse=True
        )[:5]

        return {
            "total_episodes":  total,
            "positive":        positive,
            "negative":        negative,
            "neutral":         total - positive - negative,
            "success_rate":    round(positive / total, 2) if total > 0 else 0,
            "topic_count":     len(self._patterns["outcomes"]),
            "chain_count":     len(self._patterns["chains"]),
            "top_topics":      [{"topic": k, "total": v.get("total", 0)} for k, v in top_topics],
        }